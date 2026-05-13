# waifu-scorer

A deep learning-based tool for scoring anime-style images, supporting multiple hardware backends and PyTorch environments.

## Installation

You need Python 3.10+ and pip. It is recommended to use a virtual environment.

### 1. Install PyTorch for your platform

Pick the right command for your OS / hardware from the official selector:
<https://pytorch.org/get-started/locally/>

A few common examples:

```bash
# CPU only (any OS)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU (pick the CUDA version that matches your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch --index-url https://download.pytorch.org/whl/cu128

# AMD GPU on Linux (ROCm)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# macOS (Apple Silicon, MPS backend) — just use the default PyPI wheel
pip install torch
```

### 2. Install waifu-scorer

```bash
pip install waifu-scorer
```

If you are using [uv](https://docs.astral.sh/uv/), you can let it pick a torch backend automatically:

```bash
uv pip install --torch-backend=auto waifu-scorer
```

## Usage in Python

You can also use waifu-scorer directly in your Python code:

```python
from waifu_scorer.predict import WaifuScorer

scorer = WaifuScorer()
results = scorer(["path/to/image1.jpg", "path/to/image2.png"])
for img_path, score in zip(["path/to/image1.jpg", "path/to/image2.png"], results, strict=False):
    print(f"{img_path}: {score:.3f}")
```

## Usage from Command Line

After installation, you can use the command line interface to score images:

```bash
python -m waifu_scorer path/to/image1.jpg path/to/image2.png
```

### Options

- `--model`: Path to a custom model file
- `--device`: Device to use
- `--verbose`: Enable verbose output

Example:

```bash
python -m waifu_scorer examples/waifu1.png --verbose
```

## Reference

This project refers to [waifuset](https://github.com/Eugeoter/waifuset).

---

For more details, see the code and documentation in the repository.
