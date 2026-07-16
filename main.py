"""FastAPI entry point for the geology assistant MVP."""
#
#https://files.clear.ml/GeoVision/ResNet34_UNet_clDICE_Combined.8904726077d04d40837b61ffb5958ece/artifacts/best_model_clDICE_epoch_45/best_model_clDICE_epoch_45.pth

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import logging
from math import ceil, hypot, pi
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv

from calculations import ReservoirParameters, calculate_reservoir_summary
from model_service import ModelService, ModelSettings
from visualizations import (
    image_to_data_url,
    render_binary_mask,
    render_mask,
    render_probability,
    render_well_overlay,
)
from well_planner import WellPlannerParameters, build_well_scenarios, plan_wells


PROJECT_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_PREVIEW_SIDE = 1600
MAX_PLANNING_SIDE = 1024
MAX_PLANNING_RADIUS_PX = 96
MODEL_DECISION_THRESHOLD = 0.50
MAX_AUTO_WELLS = 30
logger = logging.getLogger("uvicorn.error")

load_dotenv(PROJECT_DIR / ".env")
model_service = ModelService(ModelSettings.from_environment(PROJECT_DIR))


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="GeoVision AI",
    description="Paleochannel segmentation and screening well placement",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=PROJECT_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PROJECT_DIR / "templates" / "index.html")


@app.get("/api/status")
def api_status() -> dict:
    return model_service.status()


def _load_image(content: bytes, filename: str) -> Image.Image:
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{filename}: file is larger than 20 MB")
    try:
        image = Image.open(BytesIO(content))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"{filename}: unsupported image") from exc
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"{filename}: image is too large; use no more than 20 megapixels",
        )
    return image


def _load_edited_mask(content: bytes, target_size: tuple[int, int]) -> np.ndarray:
    try:
        mask_image = Image.open(BytesIO(content))
        mask_image.load()
        mask_image = mask_image.convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Не удалось прочитать отредактированную маску") from exc
    if mask_image.size != target_size:
        mask_image = mask_image.resize(target_size, Image.Resampling.NEAREST)
    return np.asarray(mask_image, dtype=np.uint8) >= 128


def _automatic_well_limit(mask: np.ndarray, pixel_size_m: float) -> int:
    """Estimate a safe upper bound; the greedy planner stops earlier itself."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return 1
    area_m2 = float(len(rows) * pixel_size_m**2)
    area_limit = ceil(area_m2 / (0.15 * pi * 500.0**2))
    diagonal_m = hypot(float(np.ptp(rows)), float(np.ptp(cols))) * pixel_size_m
    extent_limit = ceil(diagonal_m / 1000.0) + 2
    return min(MAX_AUTO_WELLS, max(1, area_limit, extent_limit))


def _analyze_one(
    image: Image.Image,
    filename: str,
    reservoir: ReservoirParameters,
    origin_x_m: float | None,
    origin_y_m: float | None,
    mask_override: np.ndarray | None = None,
) -> dict:
    started_at = perf_counter()
    logger.info(
        "[%s] Starting inference for %sx%s image on %s",
        filename,
        image.width,
        image.height,
        model_service.device,
    )
    probability = model_service.predict(image)
    inference_finished_at = perf_counter()
    logger.info(
        "[%s] Inference completed in %.3f s",
        filename,
        inference_finished_at - started_at,
    )
    edited = mask_override is not None
    if mask_override is None:
        # The two-class U-Net assigns the positive class itself. For its
        # softmax output this is equivalent to argmax; there is no user-facing
        # segmentation threshold.
        mask = probability >= MODEL_DECISION_THRESHOLD
        accepted_probability = probability
    else:
        mask = np.asarray(mask_override, dtype=bool)
        if mask.shape != probability.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (probability.shape[1], probability.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        # A manual geological correction is authoritative. Newly painted
        # pixels receive a neutral confidence floor, while the original U-Net
        # probability still influences ranking.
        accepted_probability = np.where(
            mask,
            np.maximum(probability, MODEL_DECISION_THRESHOLD),
            probability,
        ).astype(np.float32)
    summary = calculate_reservoir_summary(mask, accepted_probability, reservoir)
    volumetrics_finished_at = perf_counter()
    logger.info("[%s] Starting well planning", filename)

    # Well placement does not need the full display resolution. Work on a
    # coarser metric grid while preserving the same physical radius and X/Y.
    height, width = probability.shape
    radius_px = 500.0 / reservoir.pixel_size_m
    planning_factor = max(
        1,
        int(np.ceil(max(height, width) / MAX_PLANNING_SIDE)),
        int(np.ceil(radius_px / MAX_PLANNING_RADIUS_PX)),
    )
    if planning_factor > 1:
        planning_width = max(1, int(np.ceil(width / planning_factor)))
        planning_height = max(1, int(np.ceil(height / planning_factor)))
        planning_probability = cv2.resize(
            accepted_probability,
            (planning_width, planning_height),
            interpolation=cv2.INTER_AREA,
        )
        planning_mask = cv2.resize(
            mask.astype(np.uint8),
            (planning_width, planning_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        planning_probability = accepted_probability
        planning_mask = mask
    planning_pixel_size_m = reservoir.pixel_size_m * planning_factor
    auto_well_limit = _automatic_well_limit(planning_mask, planning_pixel_size_m)

    planner = WellPlannerParameters(
        pixel_size_m=planning_pixel_size_m,
        radius_m=500.0,
        max_wells=auto_well_limit,
        candidate_step_m=50.0,
        min_confidence=0.0,
        min_center_distance_m=1000.0,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        max_candidates=1200,
        endpoint_setback_m=500.0,
        min_marginal_ratio=0.10,
    )
    all_wells = plan_wells(
        mask=planning_mask,
        probability=planning_probability,
        recoverable_m3_per_pixel=(
            reservoir.recoverable_m3_per_mask_pixel * planning_factor**2
        ),
        params=planner,
    )

    # Convert display pixels back to the original image. Metric coordinates
    # were already calculated correctly on the coarser grid.
    if planning_factor > 1:
        for well in all_wells:
            well["row"] = min(
                image.height - 1,
                int(round((int(well["row"]) + 0.5) * planning_factor - 0.5)),
            )
            well["col"] = min(
                image.width - 1,
                int(round((int(well["col"]) + 0.5) * planning_factor - 0.5)),
            )
    well_scenarios = build_well_scenarios(all_wells)
    wells = well_scenarios["recommended"]["wells"]
    planning_finished_at = perf_counter()
    logger.info(
        "[%s] Well planning completed in %.3f s; wells=%s; grid factor=%s",
        filename,
        planning_finished_at - volumetrics_finished_at,
        len(all_wells),
        planning_factor,
    )

    # Sending four full-resolution PNGs can freeze the browser. Only the visual
    # preview is resized; calculations keep using the original data.
    preview_scale = min(1.0, MAX_PREVIEW_SIDE / max(image.width, image.height))
    if preview_scale < 1.0:
        preview_size = (
            max(1, int(round(image.width * preview_scale))),
            max(1, int(round(image.height * preview_scale))),
        )
        preview_image = image.resize(preview_size, Image.Resampling.LANCZOS)
        preview_probability = cv2.resize(
            probability,
            preview_size,
            interpolation=cv2.INTER_LINEAR,
        )
        preview_mask = cv2.resize(
            mask.astype(np.uint8),
            preview_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        def scale_wells(source_wells: list[dict]) -> list[dict]:
            scaled = []
            for well in source_wells:
                preview_well = dict(well)
                preview_well["row"] = int(round(int(well["row"]) * preview_scale))
                preview_well["col"] = int(round(int(well["col"]) * preview_scale))
                scaled.append(preview_well)
            return scaled

        preview_scenario_wells = {
            key: scale_wells(value["wells"]) for key, value in well_scenarios.items()
        }
        preview_pixel_size_m = reservoir.pixel_size_m / preview_scale
    else:
        preview_image = image
        preview_probability = probability
        preview_mask = mask
        preview_scenario_wells = {
            key: value["wells"] for key, value in well_scenarios.items()
        }
        preview_pixel_size_m = reservoir.pixel_size_m

    original = image_to_data_url(preview_image)
    mask_image = image_to_data_url(render_mask(preview_mask))
    mask_binary = image_to_data_url(render_binary_mask(preview_mask))
    probability_image = image_to_data_url(render_probability(preview_probability))
    scenario_images = {
        f"wells_{key}": image_to_data_url(
            render_well_overlay(
                preview_image,
                preview_mask,
                scenario_wells,
                preview_pixel_size_m,
            )
        )
        for key, scenario_wells in preview_scenario_wells.items()
    }
    visualization_finished_at = perf_counter()
    logger.info(
        "[%s] Visualization completed in %.3f s",
        filename,
        visualization_finished_at - planning_finished_at,
    )
    timings = {
        "inference": round(inference_finished_at - started_at, 3),
        "volumetrics": round(volumetrics_finished_at - inference_finished_at, 3),
        "well_planning": round(planning_finished_at - volumetrics_finished_at, 3),
        "visualization": round(visualization_finished_at - planning_finished_at, 3),
        "total": round(visualization_finished_at - started_at, 3),
    }
    logger.info("Analysis finished for %s: %s", filename, timings)

    return {
        "filename": filename,
        "width": image.width,
        "height": image.height,
        "edited": edited,
        "model_mode": model_service.mode,
        "images": {
            "original": original,
            "mask": mask_image,
            "mask_binary": mask_binary,
            "probability": probability_image,
            "wells": scenario_images["wells_recommended"],
            **scenario_images,
        },
        "summary": summary,
        "wells": wells,
        "well_scenarios": well_scenarios,
        "planner": {
            "radius_m": 500.0,
            "min_center_distance_m": 1000.0,
            "candidate_step_m": 50.0,
            "endpoint_setback_m": 500.0,
            "automatic_candidate_limit": auto_well_limit,
            "planning_downsample_factor": planning_factor,
        },
        "timings_seconds": timings,
    }


@app.post("/api/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    pixel_size_m: float = Form(10.0),
    origin_x_m: float | None = Form(None),
    origin_y_m: float | None = Form(None),
    effective_thickness_m: float = Form(10.0),
    net_to_gross: float = Form(0.70),
    porosity: float = Form(0.20),
    oil_saturation: float = Form(0.70),
    formation_volume_factor: float = Form(1.20),
    recovery_factor: float = Form(0.30),
    oil_density_t_m3: float = Form(0.85),
    oil_price_usd_bbl: float = Form(70.0),
) -> dict:
    logger.info("Received analysis request with %s file(s)", len(files))
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="A maximum of 20 images per request is supported")

    reservoir = ReservoirParameters(
        pixel_size_m=pixel_size_m,
        effective_thickness_m=effective_thickness_m,
        net_to_gross=net_to_gross,
        porosity=porosity,
        oil_saturation=oil_saturation,
        formation_volume_factor=formation_volume_factor,
        recovery_factor=recovery_factor,
        oil_density_t_m3=oil_density_t_m3,
        oil_price_usd_bbl=oil_price_usd_bbl,
    )
    try:
        reservoir.validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    results = []
    for upload in files:
        logger.info("Reading uploaded file: %s", upload.filename or "image.png")
        content = await upload.read()
        filename = upload.filename or "image.png"
        image = _load_image(content, filename)
        results.append(
            _analyze_one(
                image=image,
                filename=filename,
                reservoir=reservoir,
                origin_x_m=origin_x_m,
                origin_y_m=origin_y_m,
            )
        )

    warning = None
    if model_service.mode != "model":
        warning = (
            "Демонстрационный режим: best.pth не загружен или несовместим. "
            "Результаты нужны только для проверки интерфейса."
        )
    return {"model": model_service.status(), "warning": warning, "results": results}


@app.post("/api/recalculate")
async def recalculate(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
    pixel_size_m: float = Form(10.0),
    origin_x_m: float | None = Form(None),
    origin_y_m: float | None = Form(None),
    effective_thickness_m: float = Form(10.0),
    net_to_gross: float = Form(0.70),
    porosity: float = Form(0.20),
    oil_saturation: float = Form(0.70),
    formation_volume_factor: float = Form(1.20),
    recovery_factor: float = Form(0.30),
    oil_density_t_m3: float = Form(0.85),
    oil_price_usd_bbl: float = Form(70.0),
) -> dict:
    """Re-run calculations and well placement using a geologist-edited mask."""
    filename = file.filename or "image.png"
    image = _load_image(await file.read(), filename)
    edited_mask = _load_edited_mask(await mask.read(), image.size)
    reservoir = ReservoirParameters(
        pixel_size_m=pixel_size_m,
        effective_thickness_m=effective_thickness_m,
        net_to_gross=net_to_gross,
        porosity=porosity,
        oil_saturation=oil_saturation,
        formation_volume_factor=formation_volume_factor,
        recovery_factor=recovery_factor,
        oil_density_t_m3=oil_density_t_m3,
        oil_price_usd_bbl=oil_price_usd_bbl,
    )
    try:
        reservoir.validate()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = _analyze_one(
        image=image,
        filename=filename,
        reservoir=reservoir,
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        mask_override=edited_mask,
    )
    return {"model": model_service.status(), "warning": None, "result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
