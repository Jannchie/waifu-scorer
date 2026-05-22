import contextlib
import logging
from pathlib import Path
from typing import Any, cast, overload

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import ExifTags, Image
from safetensors.torch import load_file
from transformers import CLIPModel, CLIPProcessor

from .mlp import MLP

DEFAULT_REPO = "Eugeoter/waifu-scorer-v3"
DEFAULT_FILENAME = "model.safetensors"
logger = logging.getLogger("WaifuScorer")


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return a usable torch.device, falling back to mps/cpu when cuda is unavailable."""
    if device is not None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            logger.info("cuda not available, falling back to %s", _auto_device())
            return _auto_device()
        if requested.type == "mps" and not torch.backends.mps.is_available():
            logger.info("mps not available, falling back to cpu")
            return torch.device("cpu")
        return requested
    return _auto_device()


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_path(model_path: str | None) -> str:
    """Resolve a model identifier to a local file path.

    Accepts: None (default repo), local file, local directory (looks for model.safetensors),
    or a HuggingFace identifier in the form "user/repo" or "user/repo/filename".
    """
    if model_path is None:
        return hf_hub_download(repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME)
    p = Path(model_path)
    if p.is_file():
        return p.as_posix()
    if p.is_dir():
        return (p / DEFAULT_FILENAME).as_posix()
    user_repo, _, filename = model_path.partition("/")
    if "/" in filename:
        repo_name, _, filename = filename.partition("/")
        return hf_hub_download(repo_id=f"{user_repo}/{repo_name}", filename=filename)
    if user_repo and filename:
        return hf_hub_download(repo_id=model_path, filename=DEFAULT_FILENAME)
    msg = f"Invalid model_path: {model_path}"
    raise ValueError(msg)


def rotate_image_straight(image: Image.Image) -> Image.Image:
    with contextlib.suppress(Exception):
        if exif := image.getexif():
            orientation_tag = {v: k for k, v in ExifTags.TAGS.items()}["Orientation"]
            orientation = exif.get(orientation_tag)
            if orientation is not None and (
                degree := {
                    3: 180,
                    6: 270,
                    8: 90,
                }.get(orientation)
            ):
                image = image.rotate(degree, expand=True)
    return image


@overload
def fill_transparency(image: Image.Image, bg_color: tuple[int, int, int] = ...) -> Image.Image: ...
@overload
def fill_transparency(image: np.ndarray, bg_color: tuple[int, int, int] = ...) -> np.ndarray: ...
def fill_transparency(
    image: Image.Image | np.ndarray,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image | np.ndarray:
    r"""
    Fill the transparent part of an image with a background color.
    Please pay attention that this function doesn't change the image type.
    """
    if isinstance(image, Image.Image):
        # Only process if image has transparency
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
            alpha = image.convert("RGBA").split()[-1]

            # Create a new background image of our matt color.
            # Must be RGBA because paste requires both images have the same format
            # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
            bg = Image.new("RGBA", image.size, (*bg_color, 255))
            bg.paste(image, mask=alpha)
            return bg
        return image
    if image.shape[2] == 4:  # noqa: PLR2004
        bg = np.full_like(image, (*bg_color, 255))
        bg[:, :, :3] = image[:, :, :3]
        return bg
    return image


@overload
def convert_to_rgb(image: Image.Image, bg_color: tuple[int, int, int] = ...) -> Image.Image: ...
@overload
def convert_to_rgb(image: np.ndarray, bg_color: tuple[int, int, int] = ...) -> np.ndarray: ...
def convert_to_rgb(
    image: Image.Image | np.ndarray,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image | np.ndarray:
    r"""
    Convert an image to RGB mode and fix transparency conversion if needed.
    """
    image = fill_transparency(image, bg_color)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return image[:, :, :3]


def load_model(
    model_path: str,
    input_size: int = 768,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | str | None = None,
) -> MLP:
    model = MLP(input_size=input_size)
    state_dict = load_file(model_path) if model_path.endswith(".safetensors") else torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    if dtype is not None:
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        model = model.to(dtype=torch_dtype)
    return model


class WaifuScorer:
    def __init__(
        self,
        model_path: str | None = None,
        device: str | torch.device | None = None,
        *,
        verbose: bool = False,
        clip_model: Any = None,
        clip_processor: Any = None,
    ):
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = resolve_device(device)
        self.dtype = torch.float32

        resolved_path = resolve_model_path(model_path)
        self.logger.info("loading pretrained model from `%s`", resolved_path)
        self.mlp = load_model(resolved_path, input_size=768, device=self.device)
        if clip_model is not None and clip_processor is not None:
            self.clip = clip_model
            self.preprocess = clip_processor
        else:
            self.clip, self.preprocess = load_clip_models(device=self.device)
        self.mlp.eval()

    @torch.no_grad()
    def __call__(
        self,
        inputs: list[Image.Image | torch.Tensor | Path | str],
    ) -> list[float]:
        return self.predict(inputs)

    @torch.no_grad()
    def predict(
        self,
        inputs: list[Image.Image | torch.Tensor | Path | str],
    ) -> list[float]:
        img_embs = self.encode_inputs(inputs)
        return self.inference(img_embs)

    @torch.no_grad()
    def inference(self, img_embs: torch.Tensor) -> list[float]:
        img_embs = img_embs.to(device=self.device, dtype=self.dtype)
        predictions = self.mlp(img_embs)
        return predictions.clamp(0, 10).cpu().numpy().reshape(-1).tolist()

    def get_image(self, img_path: str | Path) -> Image.Image:
        image = Image.open(img_path)
        image = convert_to_rgb(image)
        return rotate_image_straight(image)

    def encode_inputs(
        self,
        inputs: list[Image.Image | torch.Tensor | Path | str],
    ) -> torch.Tensor:
        r"""
        Encode inputs to image embeddings.
        """
        if isinstance(inputs, (Image.Image, torch.Tensor, str, Path)):
            inputs = [inputs]

        image_or_tensors: list[Image.Image | torch.Tensor] = [self.get_image(inp) if isinstance(inp, (str, Path)) else inp for inp in inputs]
        image_idx = [i for i, img in enumerate(image_or_tensors) if isinstance(img, Image.Image)]
        batch_size = len(image_idx)
        if batch_size > 0:
            images: list[Image.Image] = [img for img in image_or_tensors if isinstance(img, Image.Image)]
            if batch_size == 1:
                images = images * 2
            img_embs = encode_images(
                images,
                self.clip,
                self.preprocess,
                device=self.device,
            )
            if batch_size == 1:
                img_embs = img_embs[:1]
            for i, idx in enumerate(image_idx):
                image_or_tensors[idx] = img_embs[i]
        tensors: list[torch.Tensor] = [t for t in image_or_tensors if isinstance(t, torch.Tensor)]
        return torch.stack(tensors, dim=0)


def load_clip_models(device: str | torch.device = "cpu") -> tuple[CLIPModel, CLIPProcessor]:
    model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name)
    if not isinstance(clip_model, CLIPModel):
        msg = f"Expected CLIPModel, got {type(clip_model).__name__}"
        raise TypeError(msg)
    clip_model = cast("CLIPModel", clip_model.to(device))  # type: ignore[arg-type]
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
    if not isinstance(processor, CLIPProcessor):
        msg = f"Expected CLIPProcessor, got {type(processor).__name__}"
        raise TypeError(msg)
    return clip_model, processor


def normalized(a: torch.Tensor, order: int = 2, dim: int = -1):
    l2 = a.norm(order, dim, keepdim=True)
    l2[l2 == 0] = 1
    return a / l2


def encode_images(
    images: list[Image.Image],
    clip_model: CLIPModel,
    preprocess: CLIPProcessor,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if isinstance(images, Image.Image):
        images = [images]
    inputs = preprocess(images=images, return_tensors="pt")  # type: ignore[call-arg]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features_out = clip_model.get_image_features(**inputs)
    # transformers 5.x changed `CLIPModel.get_image_features` to return a
    # `BaseModelOutputWithPooling` instead of the bare projected-pooled tensor
    # that 4.x returned; unwrap so downstream tensor ops keep working.
    image_features = cast("torch.Tensor", getattr(features_out, "pooler_output", features_out))
    return normalized(image_features).cpu().float()
