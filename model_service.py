"""Loading and inference for U-Net + ResNet-34 segmentation weights."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ModelSettings:
    model_path: Path
    input_size: int = 512
    encoder_name: str = "resnet34"
    in_channels: int = 3
    classes: int = 2
    positive_class_index: int = 1
    imagenet_normalization: bool = True
    allow_demo_mode: bool = True
    trust_checkpoint: bool = False
    cpu_threads: int = 4
    warmup_model: bool = True

    @classmethod
    def from_environment(cls, project_dir: Path) -> "ModelSettings":
        model_path = Path(os.getenv("MODEL_PATH", "best.pth"))
        if not model_path.is_absolute():
            model_path = project_dir / model_path
        return cls(
            model_path=model_path,
            input_size=int(os.getenv("MODEL_INPUT_SIZE", "512")),
            encoder_name=os.getenv("MODEL_ENCODER", "resnet34"),
            in_channels=int(os.getenv("MODEL_IN_CHANNELS", "3")),
            classes=int(os.getenv("MODEL_CLASSES", "2")),
            positive_class_index=int(os.getenv("POSITIVE_CLASS_INDEX", "1")),
            imagenet_normalization=os.getenv("IMAGENET_NORMALIZATION", "true").lower()
            in {"1", "true", "yes"},
            allow_demo_mode=os.getenv("ALLOW_DEMO_MODE", "true").lower()
            in {"1", "true", "yes"},
            trust_checkpoint=os.getenv("TRUST_MODEL_CHECKPOINT", "false").lower()
            in {"1", "true", "yes"},
            cpu_threads=max(1, int(os.getenv("CPU_THREADS", "4"))),
            warmup_model=os.getenv("WARMUP_MODEL", "true").lower()
            in {"1", "true", "yes"},
        )


class ModelService:
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self.model = None
        self.device = "cpu"
        self.mode = "uninitialized"
        self.error: str | None = None
        self.output_classes = settings.classes
        self.checkpoint_metadata: dict[str, object] = {}

    def load(self) -> None:
        if not self.settings.model_path.exists():
            self.mode = "demo" if self.settings.allow_demo_mode else "missing"
            self.error = f"Weights not found: {self.settings.model_path.name}"
            return

        try:
            import segmentation_models_pytorch as smp
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                torch.set_num_threads(self.settings.cpu_threads)
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    # PyTorch allows configuring inter-op threads only once.
                    pass
                if hasattr(torch.backends, "mkldnn"):
                    torch.backends.mkldnn.enabled = True
            checkpoint = self._load_checkpoint(torch)
            state_dict = self._extract_state_dict(checkpoint)
            cleaned = {}
            for key, value in state_dict.items():
                clean_key = key
                for prefix in ("module.", "model.", "net."):
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix) :]
                cleaned[clean_key] = value

            inferred_in_channels = self.settings.in_channels
            encoder_weight = cleaned.get("encoder.conv1.weight")
            if encoder_weight is not None and len(encoder_weight.shape) == 4:
                inferred_in_channels = int(encoder_weight.shape[1])

            inferred_classes = self.settings.classes
            head_weight = cleaned.get("segmentation_head.0.weight")
            if head_weight is not None and len(head_weight.shape) == 4:
                inferred_classes = int(head_weight.shape[0])

            model = smp.Unet(
                encoder_name=self.settings.encoder_name,
                encoder_weights=None,
                in_channels=inferred_in_channels,
                classes=inferred_classes,
                activation=None,
            )

            model.load_state_dict(cleaned, strict=True)
            model.to(self.device)
            if self.device == "cpu":
                model.to(memory_format=torch.channels_last)
            model.eval()
            self.model = model
            self.output_classes = inferred_classes
            self.checkpoint_metadata = self._extract_metadata(checkpoint)
            if self.settings.warmup_model:
                logger = logging.getLogger("uvicorn.error")
                logger.info(
                    "Warming up model on %s with %sx%s input...",
                    self.device,
                    self.settings.input_size,
                    self.settings.input_size,
                )
                dummy = torch.zeros(
                    1,
                    inferred_in_channels,
                    self.settings.input_size,
                    self.settings.input_size,
                    device=self.device,
                )
                if self.device == "cpu":
                    dummy = dummy.contiguous(memory_format=torch.channels_last)
                with torch.inference_mode():
                    model(dummy)
                del dummy
                logger.info("Model warmup completed.")
            self.mode = "model"
            self.error = None
        except Exception as exc:  # The status endpoint exposes a concise diagnostic.
            self.model = None
            self.mode = "demo" if self.settings.allow_demo_mode else "error"
            self.error = f"{type(exc).__name__}: {exc}"

    def _load_checkpoint(self, torch):
        """Prefer PyTorch's restricted loader and use unsafe pickle only by opt-in."""
        safe_error: Exception | None = None
        try:
            safe_types = [
                np._core.multiarray.scalar,
                np.dtype,
                type(np.dtype(np.float64)),
            ]
            with torch.serialization.safe_globals(safe_types):
                return torch.load(
                    self.settings.model_path,
                    map_location=self.device,
                    weights_only=True,
                )
        except (AttributeError, TypeError) as exc:
            safe_error = exc
        except Exception as exc:
            safe_error = exc

        if self.settings.trust_checkpoint:
            return torch.load(
                self.settings.model_path,
                map_location=self.device,
                weights_only=False,
            )

        raise RuntimeError(
            "Safe checkpoint loading failed. Run checkpoint_tool.py convert or set "
            "TRUST_MODEL_CHECKPOINT=true only for a checkpoint you trust. "
            f"Original error: {safe_error}"
        )

    @staticmethod
    def _extract_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            raise TypeError("best.pth must contain a state_dict or checkpoint dictionary")

        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
        raise KeyError("No model state_dict found in best.pth")

    @staticmethod
    def _extract_metadata(checkpoint) -> dict[str, object]:
        if not isinstance(checkpoint, dict):
            return {}
        metadata: dict[str, object] = {}
        for key in ("epoch", "start_epoch", "val_loss", "loss_type", "architecture"):
            value = checkpoint.get(key)
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        return metadata

    def status(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "model_loaded": self.mode == "model",
            "model_path": self.settings.model_path.name,
            "architecture": f"U-Net + {self.settings.encoder_name}",
            "input_size": self.settings.input_size,
            "output_classes": self.output_classes,
            "positive_class_index": self.settings.positive_class_index,
            "device": self.device,
            "cpu_threads": self.settings.cpu_threads if self.device == "cpu" else None,
            "checkpoint": self.checkpoint_metadata,
            "error": self.error,
        }

    def predict(self, image: Image.Image) -> np.ndarray:
        if self.mode == "model" and self.model is not None:
            return self._predict_model(image)
        if self.mode == "demo":
            return self._predict_demo(image)
        raise RuntimeError(self.error or "Segmentation model is not available")

    def _predict_model(self, image: Image.Image) -> np.ndarray:
        import torch

        original_width, original_height = image.size
        resized = image.convert("RGB").resize(
            (self.settings.input_size, self.settings.input_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        if self.settings.imagenet_normalization:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            array = (array - mean) / std

        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        if self.device == "cpu":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            logits = self.model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.shape[1] == 1:
                probability = torch.sigmoid(logits)[0, 0]
            else:
                class_index = self.settings.positive_class_index
                if not 0 <= class_index < logits.shape[1]:
                    raise ValueError(
                        f"POSITIVE_CLASS_INDEX={class_index} is invalid for "
                        f"{logits.shape[1]} output classes"
                    )
                probability = torch.softmax(logits, dim=1)[0, class_index]
            probability = probability.detach().cpu().numpy()

        probability = cv2.resize(
            probability,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.clip(probability, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _predict_demo(image: Image.Image) -> np.ndarray:
        """Visual smoke-test only; never presented as a trained prediction."""
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=7.0)
        local_contrast = np.abs(gray - local_background)
        saturation = rgb.max(axis=2) - rgb.min(axis=2)

        def normalize(value: np.ndarray) -> np.ndarray:
            lo, hi = np.percentile(value, [5, 95])
            return np.clip((value - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)

        probability = 0.65 * normalize(local_contrast) + 0.35 * normalize(saturation)
        probability = cv2.GaussianBlur(probability, (0, 0), sigmaX=2.0)
        return np.clip(probability, 0.0, 1.0).astype(np.float32)
