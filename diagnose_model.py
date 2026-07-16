"""Measure model loading and inference time on the current computer."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from PIL import Image

from model_service import ModelService, ModelSettings


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")
    service = ModelService(ModelSettings.from_environment(PROJECT_DIR))

    started = perf_counter()
    service.load()
    loaded = perf_counter()
    print("Status:", service.status())
    print(f"Load + warmup: {loaded - started:.3f} s")
    if not service.status()["model_loaded"]:
        raise SystemExit(1)

    if args.image:
        image = Image.open(args.image).convert("RGB")
        print(f"Image: {args.image} ({image.width}x{image.height})")
    else:
        size = service.settings.input_size
        image = Image.new("RGB", (size, size), color=(127, 127, 127))
        print(f"Image: generated {size}x{size}")

    inference_started = perf_counter()
    probability = service.predict(image)
    inference_finished = perf_counter()
    print(f"Inference: {inference_finished - inference_started:.3f} s")
    print(
        "Probability:",
        f"shape={probability.shape}",
        f"min={probability.min():.6f}",
        f"max={probability.max():.6f}",
        f"mean={probability.mean():.6f}",
    )


if __name__ == "__main__":
    main()
