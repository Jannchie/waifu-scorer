# waifu-scorer

A deep learning-based tool for scoring anime-style images, supporting multiple hardware backends and PyTorch environments.

## Installation

You need Python 3.12+ and pip. It is recommended to use a virtual environment.

```bash
pip install .[cpu]      # For CPU only
pip install .[cu121]    # For CUDA 12.1
pip install .[cu124]    # For CUDA 12.4
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
