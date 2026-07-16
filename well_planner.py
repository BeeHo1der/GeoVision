"""Greedy 2D screening algorithm for candidate well coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    convolve,
    distance_transform_edt,
    label,
    maximum_filter,
)


@dataclass(frozen=True)
class WellPlannerParameters:
    pixel_size_m: float
    radius_m: float = 500.0
    max_wells: int = 5
    candidate_step_m: float = 50.0
    min_confidence: float = 0.50
    min_center_distance_m: float | None = None
    origin_x_m: float | None = None
    origin_y_m: float | None = None
    max_candidates: int = 1200
    endpoint_setback_m: float = 500.0
    min_marginal_ratio: float = 0.10

    @property
    def spacing_m(self) -> float:
        # With non-overlapping 500 m influence circles the centers are 1000 m apart.
        return self.min_center_distance_m or self.radius_m * 2.0

    def validate(self) -> None:
        if self.pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be greater than zero")
        if self.radius_m <= 0:
            raise ValueError("radius_m must be greater than zero")
        if not 1 <= self.max_wells <= 100:
            raise ValueError("max_wells must be between 1 and 100")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.endpoint_setback_m < 0:
            raise ValueError("endpoint_setback_m must be non-negative")
        if not 0 <= self.min_marginal_ratio <= 1:
            raise ValueError("min_marginal_ratio must be between 0 and 1")
        if not 50 <= self.max_candidates <= 10_000:
            raise ValueError("max_candidates must be between 50 and 10000")


def _disk_slices(
    row: int,
    col: int,
    radius_px: int,
    height: int,
    width: int,
) -> tuple[slice, slice, np.ndarray]:
    r0 = max(0, row - radius_px)
    r1 = min(height, row + radius_px + 1)
    c0 = max(0, col - radius_px)
    c1 = min(width, col + radius_px + 1)
    yy, xx = np.ogrid[r0:r1, c0:c1]
    disk = (yy - row) ** 2 + (xx - col) ** 2 <= radius_px**2
    return slice(r0, r1), slice(c0, c1), disk


def _sample_centerline(
    skeleton: np.ndarray,
    center_distance: np.ndarray,
    step_px: int,
    endpoint_distance: np.ndarray | None = None,
    endpoint_setback_px: float = 0.0,
) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(skeleton)
    if len(rows) == 0:
        return []

    # Keep the most central skeleton pixel inside every sampling cell.
    sampled: dict[tuple[int, int], tuple[int, int, float]] = {}
    for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
        if (
            endpoint_distance is not None
            and float(endpoint_distance[row, col]) < endpoint_setback_px
        ):
            continue
        key = (row // step_px, col // step_px)
        distance = float(center_distance[row, col])
        current = sampled.get(key)
        if current is None or distance > current[2]:
            sampled[key] = (row, col, distance)
    return [(row, col) for row, col, _ in sampled.values()]


def _clean_mask(mask: np.ndarray, min_object_px: int) -> np.ndarray:
    components, _ = label(mask)
    component_sizes = np.bincount(components.ravel())
    keep = component_sizes >= min_object_px
    if keep.size:
        keep[0] = False
    cleaned = keep[components]

    filled = binary_fill_holes(cleaned)
    holes = filled & ~cleaned
    hole_labels, _ = label(holes)
    hole_sizes = np.bincount(hole_labels.ravel())
    fill_small = hole_sizes < min_object_px
    if fill_small.size:
        fill_small[0] = False
    return cleaned | fill_small[hole_labels]


def _centerline_ridge(mask: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Return a dependency-light approximation of a medial centerline."""
    local_maximum = maximum_filter(distance, size=3, mode="constant")
    ridge = mask & (distance > 0) & (distance >= local_maximum - 1e-6)
    if ridge.any():
        return ridge
    # This fallback is practically unreachable, but keeps tiny valid objects usable.
    ridge = np.zeros_like(mask, dtype=bool)
    if mask.any():
        row, col = np.unravel_index(np.argmax(distance), distance.shape)
        ridge[row, col] = True
    return ridge


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Create a one-pixel-wide centerline without an extra skimage dependency."""
    working = np.asarray(mask, dtype=bool).copy()
    skeleton = np.zeros_like(working)
    element = np.asarray(
        [[False, True, False], [True, True, True], [False, True, False]],
        dtype=bool,
    )
    max_iterations = max(mask.shape)

    for _ in range(max_iterations):
        if not working.any():
            break
        opened = binary_opening(working, structure=element)
        skeleton |= working & ~opened
        working = binary_erosion(working, structure=element)
    return skeleton


def _endpoint_distance(skeleton: np.ndarray) -> np.ndarray | None:
    """Return Euclidean distance to terminal centerline pixels."""
    if not skeleton.any():
        return None
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = convolve(
        skeleton.astype(np.uint8),
        kernel,
        mode="constant",
        cval=0,
    ) - skeleton.astype(np.uint8)
    endpoints = skeleton & (neighbor_count <= 1)
    if not endpoints.any():
        return None
    return distance_transform_edt(~endpoints)


def build_well_scenarios(
    wells: list[dict[str, float | int]],
) -> dict[str, dict[str, float | int | list[dict[str, float | int]]]]:
    """Split a ranked well plan into minimum, balanced and maximum scenarios.

    The thresholds refer to the cumulative expected volume available to the
    complete ranked plan. This makes the number of wells adapt to each mask.
    """
    targets = {
        "minimum": ("Минимум", 0.55),
        "recommended": ("Оптимально", 0.80),
        "maximum": ("Максимум", 0.95),
    }
    if not wells:
        return {
            key: {
                "label": label_text,
                "target_share": target,
                "count": 0,
                "captured_share": 0.0,
                "volume_m3": 0.0,
                "wells": [],
            }
            for key, (label_text, target) in targets.items()
        }

    marginal = np.asarray(
        [float(well["marginal_volume_m3"]) for well in wells],
        dtype=np.float64,
    )
    cumulative = np.cumsum(marginal)
    achievable = max(float(cumulative[-1]), 1e-9)
    raw_minimum = int(np.searchsorted(cumulative / achievable, 0.55, side="left") + 1)
    raw_recommended = int(np.searchsorted(cumulative / achievable, 0.80, side="left") + 1)
    if len(wells) >= 3:
        minimum_count = min(raw_minimum, len(wells) - 2)
        recommended_count = min(
            len(wells) - 1,
            max(minimum_count + 1, raw_recommended),
        )
    elif len(wells) == 2:
        minimum_count = 1
        recommended_count = 2
    else:
        minimum_count = recommended_count = 1
    scenario_counts = {
        "minimum": minimum_count,
        "recommended": recommended_count,
        "maximum": len(wells),
    }

    scenarios: dict[
        str, dict[str, float | int | list[dict[str, float | int]]]
    ] = {}
    for key, (label_text, target) in targets.items():
        count = scenario_counts[key]
        captured_volume = float(cumulative[count - 1]) if count else 0.0
        scenarios[key] = {
            "label": label_text,
            "target_share": target,
            "count": count,
            "captured_share": round(captured_volume / achievable, 4),
            "volume_m3": round(captured_volume, 2),
            "wells": wells[:count],
        }
    return scenarios


def plan_wells(
    mask: np.ndarray,
    probability: np.ndarray,
    recoverable_m3_per_pixel: float,
    params: WellPlannerParameters,
    existing_wells_xy_m: list[tuple[float, float]] | None = None,
) -> list[dict[str, float | int]]:
    """Select well targets by maximizing newly covered expected volume.

    Every selected well covers a circle with ``radius_m``. By default circles do
    not overlap, so the center-to-center spacing is ``2 * radius_m``.
    """
    params.validate()
    existing_wells_xy_m = existing_wells_xy_m or []
    probability = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    raw_mask = np.asarray(mask, dtype=bool)
    if raw_mask.shape != probability.shape:
        raise ValueError("mask and probability must have the same shape")

    # Remove only tiny pixel artefacts; preserve elongated channel geometry.
    min_object_px = max(4, int(round(400 / (params.pixel_size_m**2))))
    clean_mask = _clean_mask(raw_mask, min_object_px)
    if not clean_mask.any():
        return []

    center_distance = distance_transform_edt(clean_mask)
    skeleton = _morphological_skeleton(clean_mask)
    if not skeleton.any():
        skeleton = _centerline_ridge(clean_mask, center_distance)
    endpoint_distance = _endpoint_distance(skeleton)
    step_px = max(1, int(round(params.candidate_step_m / params.pixel_size_m)))
    candidates = _sample_centerline(
        skeleton,
        center_distance,
        step_px,
        endpoint_distance=endpoint_distance,
        endpoint_setback_px=params.endpoint_setback_m / params.pixel_size_m,
    )
    if not candidates and endpoint_distance is not None:
        # A very short isolated body may not have 500 m of interior length.
        # Keep its central target usable instead of returning an empty plan.
        candidates = _sample_centerline(skeleton, center_distance, step_px)
    if not candidates:
        return []
    if len(candidates) > params.max_candidates:
        # Keep a spatially uniform subset. This prevents noisy/full masks from
        # turning the greedy circle evaluation into billions of pixel checks.
        candidates.sort()
        indices = np.linspace(
            0,
            len(candidates) - 1,
            params.max_candidates,
            dtype=np.int64,
        )
        candidates = [candidates[int(index)] for index in indices]

    radius_px = max(1, int(round(params.radius_m / params.pixel_size_m)))
    height, width = clean_mask.shape
    expected_volume_map = (
        clean_mask.astype(np.float32)
        * probability
        * float(recoverable_m3_per_pixel)
    )
    covered = np.zeros_like(clean_mask, dtype=bool)
    selected: list[dict[str, float | int]] = []
    first_marginal_volume: float | None = None
    max_center_distance = max(float(center_distance.max()), 1.0)

    def pixel_to_xy(row: int, col: int) -> tuple[float, float]:
        local_x = (col + 0.5) * params.pixel_size_m
        local_y = (row + 0.5) * params.pixel_size_m
        x = local_x if params.origin_x_m is None else params.origin_x_m + local_x
        # A supplied origin is treated as the upper-left northing of a north-up map.
        y = local_y if params.origin_y_m is None else params.origin_y_m - local_y
        return x, y

    def is_far_enough(x: float, y: float) -> bool:
        for well in selected:
            if hypot(x - float(well["x_m"]), y - float(well["y_m"])) < params.spacing_m:
                return False
        for existing_x, existing_y in existing_wells_xy_m:
            if hypot(x - existing_x, y - existing_y) < params.spacing_m:
                return False
        return True

    for _ in range(params.max_wells):
        evaluated: list[dict[str, float | int | tuple[slice, slice, np.ndarray]]] = []
        for row, col in candidates:
            x_m, y_m = pixel_to_xy(row, col)
            if not is_far_enough(x_m, y_m):
                continue

            rs, cs, disk = _disk_slices(row, col, radius_px, height, width)
            local_mask = clean_mask[rs, cs] & disk
            if not local_mask.any():
                continue

            local_probability = probability[rs, cs][local_mask]
            mean_probability = float(local_probability.mean())
            if mean_probability < params.min_confidence:
                continue

            new_pixels = local_mask & ~covered[rs, cs]
            marginal_volume = float(expected_volume_map[rs, cs][new_pixels].sum())
            marginal_area_m2 = float(new_pixels.sum() * params.pixel_size_m**2)
            if marginal_volume <= 0:
                continue

            evaluated.append(
                {
                    "row": row,
                    "col": col,
                    "x_m": x_m,
                    "y_m": y_m,
                    "mean_probability": mean_probability,
                    "marginal_volume_m3": marginal_volume,
                    "marginal_area_m2": marginal_area_m2,
                    "boundary_distance_m": float(center_distance[row, col] * params.pixel_size_m),
                    "centrality": float(center_distance[row, col] / max_center_distance),
                    "disk_data": (rs, cs, disk),
                }
            )

        if not evaluated:
            break

        max_volume = max(float(item["marginal_volume_m3"]) for item in evaluated)
        for item in evaluated:
            volume_score = float(item["marginal_volume_m3"]) / max(max_volume, 1e-9)
            item["score"] = (
                0.70 * volume_score
                + 0.20 * float(item["mean_probability"])
                + 0.10 * float(item["centrality"])
            )

        best = max(evaluated, key=lambda item: float(item["score"]))
        marginal_volume = float(best["marginal_volume_m3"])
        if first_marginal_volume is None:
            first_marginal_volume = marginal_volume
        elif marginal_volume < first_marginal_volume * params.min_marginal_ratio:
            break

        rs, cs, disk = best.pop("disk_data")  # type: ignore[assignment]
        covered_region = covered[rs, cs]
        covered_region[disk] = True
        covered[rs, cs] = covered_region
        best.pop("centrality", None)
        best["rank"] = len(selected) + 1
        best["radius_m"] = params.radius_m
        best["min_center_distance_m"] = params.spacing_m
        best["score"] = round(float(best["score"]), 4)
        best["mean_probability"] = round(float(best["mean_probability"]), 4)
        best["marginal_volume_m3"] = round(marginal_volume, 2)
        best["marginal_area_m2"] = round(float(best["marginal_area_m2"]), 2)
        best["boundary_distance_m"] = round(float(best["boundary_distance_m"]), 2)
        best["x_m"] = round(float(best["x_m"]), 2)
        best["y_m"] = round(float(best["y_m"]), 2)
        selected.append(best)  # type: ignore[arg-type]

    return selected
