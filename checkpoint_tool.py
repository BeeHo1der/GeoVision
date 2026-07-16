"""Inspect and convert trusted PyTorch training checkpoints for inference."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np


def load_checkpoint(path: Path, trust: bool):
    import torch

    safe_error: Exception | None = None
    try:
        safe_types = [
            np._core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(safe_types):
            return torch.load(path, map_location="cpu", weights_only=True), "safe"
    except Exception as exc:
        safe_error = exc

    if not trust:
        raise RuntimeError(
            "Restricted loading failed. If this is your own trusted checkpoint, "
            "repeat the command with --trust. Never do this for an unknown file. "
            f"Original error: {safe_error}"
        )
    return torch.load(path, map_location="cpu", weights_only=False), "trusted-pickle"


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    for key in ("model_state_dict", "state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
        return checkpoint
    raise KeyError("No model state_dict found")


def to_plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def model_facts(state_dict) -> dict[str, object]:
    encoder_weight = state_dict.get("encoder.conv1.weight")
    head_weight = state_dict.get("segmentation_head.0.weight")
    return {
        "tensor_count": len(state_dict),
        "in_channels": int(encoder_weight.shape[1]) if encoder_weight is not None else None,
        "classes": int(head_weight.shape[0]) if head_weight is not None else None,
        "segmentation_kernel": list(head_weight.shape[2:]) if head_weight is not None else None,
    }


def inspect_checkpoint(path: Path, trust: bool) -> None:
    checkpoint, load_mode = load_checkpoint(path, trust)
    state_dict = extract_state_dict(checkpoint)
    facts = model_facts(state_dict)
    print(f"File: {path.resolve()}")
    print(f"Size: {path.stat().st_size / 1024 / 1024:.2f} MiB")
    print(f"Load mode: {load_mode}")
    print(f"Top-level keys: {list(checkpoint.keys())}")
    print(f"Model tensors: {facts['tensor_count']}")
    print(f"Input channels: {facts['in_channels']}")
    print(f"Output classes: {facts['classes']}")
    print(f"Segmentation kernel: {facts['segmentation_kernel']}")
    for key in ("epoch", "start_epoch", "val_loss", "loss_type", "val_metrics"):
        if key in checkpoint:
            print(f"{key}: {to_plain(checkpoint[key])}")
    optimizer = checkpoint.get("optimizer_state_dict")
    print(f"Contains optimizer state: {isinstance(optimizer, dict)}")


def convert_checkpoint(source: Path, destination: Path, trust: bool) -> None:
    import torch

    checkpoint, load_mode = load_checkpoint(source, trust)
    state_dict = extract_state_dict(checkpoint)
    cpu_state = OrderedDict(
        (key, tensor.detach().cpu()) for key, tensor in state_dict.items()
    )
    facts = model_facts(cpu_state)
    converted = {
        "format_version": 1,
        "model_state_dict": cpu_state,
        "architecture": "segmentation_models_pytorch.Unet",
        "encoder_name": "resnet34",
        "encoder_pretraining": "imagenet",
        "input_size": 512,
        "input_channels": facts["in_channels"],
        "classes": facts["classes"],
        "positive_class_index": 1,
        "normalization": "imagenet",
        "source_epoch": to_plain(checkpoint.get("epoch")),
        "source_val_loss": to_plain(checkpoint.get("val_loss")),
        "source_val_metrics": to_plain(checkpoint.get("val_metrics", {})),
        "source_loss_type": to_plain(checkpoint.get("loss_type")),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, destination)
    print(f"Loaded with: {load_mode}")
    print(f"Saved inference checkpoint: {destination.resolve()}")
    print(f"New size: {destination.stat().st_size / 1024 / 1024:.2f} MiB")
    print("Optimizer state was removed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print checkpoint metadata")
    inspect_parser.add_argument("checkpoint", type=Path)
    inspect_parser.add_argument("--trust", action="store_true")

    convert_parser = subparsers.add_parser("convert", help="Create inference-only checkpoint")
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("destination", type=Path)
    convert_parser.add_argument("--trust", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        inspect_checkpoint(args.checkpoint, args.trust)
    else:
        convert_checkpoint(args.source, args.destination, args.trust)


if __name__ == "__main__":
    main()
