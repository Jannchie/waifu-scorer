import argparse

from waifu_scorer.predict import WaifuScorer


def main():
    parser = argparse.ArgumentParser(description="WaifuScorer: Score images using a pretrained model.")
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s) to score.")
    parser.add_argument("--model", type=str, default=None, help="Path to model file (optional, default: auto download)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    scorer = WaifuScorer(
        model_path=args.model,
        device=args.device,
        verbose=args.verbose,
    )

    results = scorer(args.images)
    for img_path, score in zip(args.images, results, strict=False):
        print(f"{img_path}: {score:.3f}")  # noqa: T201


if __name__ == "__main__":
    main()
