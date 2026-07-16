"""Volumetric and high-level economic calculations for the MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


M3_TO_BBL = 6.28981077


@dataclass(frozen=True)
class ReservoirParameters:
    pixel_size_m: float = 10.0
    effective_thickness_m: float = 10.0
    net_to_gross: float = 0.70
    porosity: float = 0.20
    oil_saturation: float = 0.70
    formation_volume_factor: float = 1.20
    recovery_factor: float = 0.30
    oil_density_t_m3: float = 0.85
    oil_price_usd_bbl: float = 70.0

    def validate(self) -> None:
        positive = {
            "pixel_size_m": self.pixel_size_m,
            "effective_thickness_m": self.effective_thickness_m,
            "formation_volume_factor": self.formation_volume_factor,
            "oil_density_t_m3": self.oil_density_t_m3,
            "oil_price_usd_bbl": self.oil_price_usd_bbl,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        fractions = {
            "net_to_gross": self.net_to_gross,
            "porosity": self.porosity,
            "oil_saturation": self.oil_saturation,
            "recovery_factor": self.recovery_factor,
        }
        for name, value in fractions.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

    @property
    def recoverable_m3_per_mask_pixel(self) -> float:
        """Recoverable stock-tank oil represented by one positive mask pixel."""
        self.validate()
        pixel_area_m2 = self.pixel_size_m**2
        oil_in_place_m3 = (
            pixel_area_m2
            * self.effective_thickness_m
            * self.net_to_gross
            * self.porosity
            * self.oil_saturation
            / self.formation_volume_factor
        )
        return oil_in_place_m3 * self.recovery_factor


def calculate_reservoir_summary(
    mask: np.ndarray,
    probability: np.ndarray,
    params: ReservoirParameters,
) -> dict[str, float | dict[str, float]]:
    """Calculate deterministic and probability-weighted volumetrics.

    These values are a screening estimate, not an audited reserves statement.
    """
    params.validate()
    mask_bool = np.asarray(mask, dtype=bool)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    if mask_bool.shape != probability.shape:
        raise ValueError("mask and probability must have the same shape")

    pixel_area_m2 = params.pixel_size_m**2
    mask_pixels = int(mask_bool.sum())
    probability_mass = float(probability[mask_bool].sum())

    area_m2 = mask_pixels * pixel_area_m2
    expected_area_m2 = probability_mass * pixel_area_m2
    gross_rock_volume_m3 = area_m2 * params.effective_thickness_m

    oil_in_place_m3 = (
        gross_rock_volume_m3
        * params.net_to_gross
        * params.porosity
        * params.oil_saturation
        / params.formation_volume_factor
    )
    recoverable_m3 = oil_in_place_m3 * params.recovery_factor

    expected_recoverable_m3 = (
        probability_mass * params.recoverable_m3_per_mask_pixel
    )
    expected_tonnes = expected_recoverable_m3 * params.oil_density_t_m3
    expected_barrels = expected_recoverable_m3 * M3_TO_BBL
    recoverable_barrels = recoverable_m3 * M3_TO_BBL
    gross_value_usd = recoverable_barrels * params.oil_price_usd_bbl
    expected_gross_value_usd = expected_barrels * params.oil_price_usd_bbl

    mean_probability = (
        float(probability[mask_bool].mean()) if mask_pixels else 0.0
    )

    return {
        "parameters": asdict(params),
        "mask_area_m2": round(area_m2, 2),
        "mask_area_km2": round(area_m2 / 1_000_000, 4),
        "probability_weighted_area_m2": round(expected_area_m2, 2),
        "mean_probability": round(mean_probability, 4),
        "gross_rock_volume_m3": round(gross_rock_volume_m3, 2),
        "oil_in_place_m3": round(oil_in_place_m3, 2),
        "recoverable_m3": round(recoverable_m3, 2),
        "expected_recoverable_m3": round(expected_recoverable_m3, 2),
        "expected_tonnes": round(expected_tonnes, 2),
        "expected_barrels": round(expected_barrels, 2),
        "gross_value_usd": round(gross_value_usd, 2),
        "expected_gross_value_usd": round(expected_gross_value_usd, 2),
    }
