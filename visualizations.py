"""PNG renderers used by the browser UI."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def image_to_data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    payload = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def render_mask(mask: np.ndarray) -> Image.Image:
    height, width = mask.shape
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[:] = (8, 18, 42)
    result[np.asarray(mask, dtype=bool)] = (42, 105, 255)
    return Image.fromarray(result, mode="RGB")


def render_binary_mask(mask: np.ndarray) -> Image.Image:
    """Binary transport image used by the browser mask editor."""
    return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")


def render_probability(probability: np.ndarray) -> Image.Image:
    """Render absolute U-Net confidence: blue=low, red=high."""
    values = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    stops = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
    colors = np.asarray(
        [
            (7, 26, 82),
            (28, 85, 196),
            (211, 222, 242),
            (229, 105, 92),
            (153, 9, 23),
        ],
        dtype=np.float32,
    )
    flat = values.ravel()
    channels = [np.interp(flat, stops, colors[:, index]) for index in range(3)]
    rgb = np.stack(channels, axis=1).reshape((*values.shape, 3)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def render_well_overlay(
    image: Image.Image,
    mask: np.ndarray,
    wells: list[dict[str, float | int]],
    pixel_size_m: float,
) -> Image.Image:
    base = image.convert("RGBA")
    mask_layer = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_layer[np.asarray(mask, dtype=bool)] = (42, 105, 255, 72)
    base = Image.alpha_composite(base, Image.fromarray(mask_layer, mode="RGBA"))
    draw = ImageDraw.Draw(base)
    font = ImageFont.load_default()

    for well in wells:
        col = int(well["col"])
        row = int(well["row"])
        radius_px = max(1, int(round(float(well["radius_m"]) / pixel_size_m)))
        rank = int(well["rank"])
        draw.ellipse(
            (col - radius_px, row - radius_px, col + radius_px, row + radius_px),
            outline=(20, 225, 170, 230),
            width=max(2, round(min(base.size) / 250)),
        )
        point_radius = max(5, round(min(base.size) / 90))
        draw.ellipse(
            (col - point_radius, row - point_radius, col + point_radius, row + point_radius),
            fill=(255, 255, 255, 255),
            outline=(18, 89, 240, 255),
            width=3,
        )
        label = str(rank)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (col - (box[2] - box[0]) / 2, row - (box[3] - box[1]) / 2 - 1),
            label,
            font=font,
            fill=(8, 18, 42, 255),
        )
    return base.convert("RGB")
