from math import hypot

import numpy as np

from calculations import ReservoirParameters, calculate_reservoir_summary
from well_planner import WellPlannerParameters, build_well_scenarios, plan_wells


def test_reservoir_calculation_is_probability_weighted():
    mask = np.ones((2, 2), dtype=bool)
    probability = np.full((2, 2), 0.5, dtype=np.float32)
    params = ReservoirParameters(
        pixel_size_m=10,
        effective_thickness_m=10,
        net_to_gross=1,
        porosity=1,
        oil_saturation=1,
        formation_volume_factor=1,
        recovery_factor=1,
        oil_density_t_m3=1,
        oil_price_usd_bbl=1,
    )

    summary = calculate_reservoir_summary(mask, probability, params)

    assert summary["mask_area_m2"] == 400
    assert summary["recoverable_m3"] == 4000
    assert summary["expected_recoverable_m3"] == 2000


def test_well_centers_respect_two_radii_spacing():
    mask = np.zeros((120, 600), dtype=bool)
    mask[45:75, 10:590] = True
    probability = np.zeros_like(mask, dtype=np.float32)
    probability[mask] = 0.9
    params = WellPlannerParameters(
        pixel_size_m=10,
        radius_m=500,
        max_wells=4,
        candidate_step_m=50,
        min_confidence=0.5,
    )

    wells = plan_wells(
        mask,
        probability,
        recoverable_m3_per_pixel=1.0,
        params=params,
    )

    assert params.spacing_m == 1000
    assert len(wells) >= 2
    for index, first in enumerate(wells):
        for second in wells[index + 1 :]:
            distance = hypot(first["x_m"] - second["x_m"], first["y_m"] - second["y_m"])
            assert distance >= 1000


def test_automatic_scenarios_are_ordered_and_use_all_qualified_wells():
    wells = [
        {"marginal_volume_m3": 50.0},
        {"marginal_volume_m3": 30.0},
        {"marginal_volume_m3": 15.0},
        {"marginal_volume_m3": 5.0},
    ]
    scenarios = build_well_scenarios(wells)

    assert scenarios["minimum"]["count"] < scenarios["recommended"]["count"]
    assert scenarios["recommended"]["count"] < scenarios["maximum"]["count"]
    assert scenarios["maximum"]["count"] == len(wells)
