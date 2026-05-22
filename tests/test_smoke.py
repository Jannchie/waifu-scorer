from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from waifu_scorer import predict
from waifu_scorer.mlp import MLP
from waifu_scorer.predict import (
    DEFAULT_FILENAME,
    DEFAULT_REPO,
    convert_to_rgb,
    fill_transparency,
    normalized,
    resolve_device,
    resolve_model_path,
    rotate_image_straight,
)


def test_mlp_forward_shape():
    model = MLP(input_size=768)
    model.eval()
    out = model(torch.randn(4, 768))
    assert out.shape == (4, 1)


def test_resolve_device_auto_returns_cpu_when_nothing_available():
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.backends.mps, "is_available", return_value=False),
    ):
        assert resolve_device(None).type == "cpu"


def test_resolve_device_falls_back_when_cuda_requested_but_unavailable():
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.backends.mps, "is_available", return_value=False),
    ):
        assert resolve_device("cuda").type == "cpu"


def test_resolve_device_respects_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_model_path_local_file(tmp_path: Path):
    f = tmp_path / "weights.safetensors"
    f.write_bytes(b"")
    assert resolve_model_path(str(f)) == f.as_posix()


def test_resolve_model_path_local_dir(tmp_path: Path):
    assert resolve_model_path(str(tmp_path)) == (tmp_path / DEFAULT_FILENAME).as_posix()


def test_resolve_model_path_default_calls_hub(tmp_path: Path):
    fake = tmp_path / "downloaded.safetensors"
    fake.write_bytes(b"")
    with patch.object(predict, "hf_hub_download", return_value=fake.as_posix()) as mock:
        assert resolve_model_path(None) == fake.as_posix()
        mock.assert_called_once_with(repo_id=DEFAULT_REPO, filename=DEFAULT_FILENAME)


def test_resolve_model_path_user_repo_calls_hub(tmp_path: Path):
    fake = tmp_path / "downloaded.safetensors"
    fake.write_bytes(b"")
    with patch.object(predict, "hf_hub_download", return_value=fake.as_posix()) as mock:
        resolve_model_path("user/repo")
        mock.assert_called_once_with(repo_id="user/repo", filename=DEFAULT_FILENAME)


def test_resolve_model_path_user_repo_filename_calls_hub(tmp_path: Path):
    fake = tmp_path / "downloaded.safetensors"
    fake.write_bytes(b"")
    with patch.object(predict, "hf_hub_download", return_value=fake.as_posix()) as mock:
        resolve_model_path("user/repo/custom.safetensors")
        mock.assert_called_once_with(repo_id="user/repo", filename="custom.safetensors")


def test_resolve_model_path_invalid_raises():
    with pytest.raises(ValueError, match="Invalid model_path"):
        resolve_model_path("not-a-valid-identifier")


def test_convert_to_rgb_strips_alpha():
    img = Image.new("RGBA", (8, 8), (10, 20, 30, 0))
    out = convert_to_rgb(img)
    assert out.mode == "RGB"
    assert out.size == (8, 8)


def test_fill_transparency_numpy_with_alpha():
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    out = fill_transparency(arr, bg_color=(1, 2, 3))
    assert out.shape == (4, 4, 4)


def test_rotate_image_straight_no_exif_passthrough():
    img = Image.new("RGB", (4, 6))
    out = rotate_image_straight(img)
    assert out.size == (4, 6)


def test_normalized_unit_norm():
    x = torch.tensor([[3.0, 4.0]])
    out = normalized(x)
    assert torch.allclose(out.norm(dim=-1), torch.ones(1))


def test_normalized_handles_zero_vector():
    x = torch.zeros(1, 4)
    out = normalized(x)
    assert torch.equal(out, x)
