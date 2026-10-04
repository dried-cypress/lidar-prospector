from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union


@dataclass(frozen=True, slots=True)
class TerrainCandidate:
    candidate_id: int
    geometry: Any
    score: float
    area_m2: float
    relief_m: float
    polarity: str
    strongest_scale_m: float
    lidar_score: float = 0.0
    persistence_score: float = 0.0
    morphology_score: float = 0.0
    modern_penalty: float = 0.0
    satellite_support: float = 0.0
    he_similarity: float = 0.0
    classification: str = "candidate"
    reasons: tuple[str, ...] = ()
    morphology: tuple[tuple[str, float], ...] = ()
    linear_score: float = 0.0


def _fill_invalid(data: np.ndarray) -> np.ndarray:
    if np.isfinite(data).all():
        return data.astype("float32", copy=False)

    from scipy.ndimage import distance_transform_edt

    valid = np.isfinite(data)
    if not valid.any():
        raise ValueError("DTM contains no usable elevation values")
    if valid.all():
        return data.astype("float32", copy=False)

    _, indices = distance_transform_edt(
        ~valid,
        return_distances=True,
        return_indices=True,
    )
    filled = data.copy()
    invalid_y = indices[0][~valid]
    invalid_x = indices[1][~valid]
    filled[~valid] = data[invalid_y, invalid_x]
    return filled.astype("float32", copy=False)


def calculate_hillshade(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    *,
    azimuth_degrees: float = 315.0,
    altitude_degrees: float = 45.0,
) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    elevation = _fill_invalid(elevation)
    smoothed = gaussian_filter(elevation, sigma=0.6)
    dy, dx = np.gradient(smoothed, y_resolution, x_resolution)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azimuth = np.deg2rad(azimuth_degrees)
    altitude = np.deg2rad(altitude_degrees)
    shaded = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    return np.clip((shaded + 1.0) * 127.5, 0.0, 255.0).astype("uint8")


def calculate_multidirectional_hillshade(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    *,
    azimuths: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
    altitude_degrees: float = 35.0,
) -> np.ndarray:
    shades = [
        calculate_hillshade(
            elevation,
            x_resolution,
            y_resolution,
            azimuth_degrees=azimuth,
            altitude_degrees=altitude_degrees,
        ).astype("float32")
        for azimuth in azimuths
    ]
    return np.mean(shades, axis=0).astype("uint8")


def calculate_archaeological_hillshade(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
) -> np.ndarray:
    """Create the locked v0.2.1 archaeological hillshade presentation."""
    elevation = _fill_invalid(elevation)
    multi = calculate_multidirectional_hillshade(
        elevation,
        x_resolution,
        y_resolution,
        azimuths=(0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0),
        altitude_degrees=38.0,
    ).astype("float32")
    directional = calculate_hillshade(
        elevation,
        x_resolution,
        y_resolution,
        azimuth_degrees=315.0,
        altitude_degrees=42.0,
    ).astype("float32")
    blended = 0.70 * multi + 0.30 * directional
    finite = blended[np.isfinite(blended)]
    if finite.size == 0:
        return np.zeros(blended.shape, dtype="uint8")
    low, high = np.percentile(finite, (2.0, 98.0))
    if not high > low:
        return np.clip(blended, 0, 255).astype("uint8")
    stretched = np.clip((blended - low) / (high - low), 0.0, 1.0)
    stretched = np.power(stretched, 0.90)
    return np.round(stretched * 255.0).astype("uint8")


@dataclass(frozen=True, slots=True)
class SensitivityProfile:
    name: str
    relief_scales_m: tuple[float, ...]
    threshold_percentile: float
    linear_percentile: float
    hough_percentile: float
    min_line_length_m: float
    min_area_m2: float
    max_area_m2: float
    max_candidates: int
    detailed_buffer_m: float
    monument_buffer_m: float
    closing_pixels: int


SENSITIVITY_PROFILES: dict[str, SensitivityProfile] = {
    "low": SensitivityProfile(
        name="low",
        relief_scales_m=(12.0, 24.0, 48.0, 64.0),
        threshold_percentile=99.70,
        linear_percentile=99.10,
        hough_percentile=98.90,
        min_line_length_m=35.0,
        min_area_m2=24.0,
        max_area_m2=18000.0,
        max_candidates=50,
        detailed_buffer_m=5.0,
        monument_buffer_m=3.0,
        closing_pixels=2,
    ),
    "medium": SensitivityProfile(
        name="medium",
        relief_scales_m=(4.0, 8.0, 16.0, 32.0, 64.0),
        threshold_percentile=99.25,
        linear_percentile=97.25,
        hough_percentile=97.25,
        min_line_length_m=20.0,
        min_area_m2=8.0,
        max_area_m2=12000.0,
        max_candidates=100,
        detailed_buffer_m=4.0,
        monument_buffer_m=2.0,
        closing_pixels=2,
    ),
    "high": SensitivityProfile(
        name="high",
        relief_scales_m=(2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
        threshold_percentile=97.50,
        linear_percentile=94.50,
        hough_percentile=95.00,
        min_line_length_m=12.0,
        min_area_m2=4.0,
        max_area_m2=14000.0,
        max_candidates=250,
        detailed_buffer_m=4.0,
        monument_buffer_m=1.5,
        closing_pixels=2,
    ),
}


def _resolve_workers(requested: int, task_count: int, pixels: int) -> int:
    if requested > 0:
        return max(1, min(requested, task_count))
    if task_count <= 1 or pixels < 250_000:
        return 1
    import os
    return max(1, min(task_count, min(4, (os.cpu_count() or 2))))


def _local_relief_for_scale(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    scale_m: float,
) -> tuple[float, np.ndarray]:
    from scipy.ndimage import gaussian_filter

    sigma_x = max(0.8, scale_m / (2.355 * x_resolution))
    sigma_y = max(0.8, scale_m / (2.355 * y_resolution))
    local_mean = gaussian_filter(elevation, sigma=(sigma_y, sigma_x), mode="nearest")
    return scale_m, (elevation - local_mean).astype("float32")


def calculate_local_relief_models(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    scales_m: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0),
    *,
    workers: int = 0,
) -> dict[float, np.ndarray]:
    """Calculate multiple local-relief models, optionally in parallel."""
    elevation = _fill_invalid(elevation)
    resolved_workers = _resolve_workers(workers, len(scales_m), elevation.size)
    if resolved_workers == 1:
        results = [
            _local_relief_for_scale(elevation, x_resolution, y_resolution, scale_m)
            for scale_m in scales_m
        ]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="lrm") as pool:
            results = list(
                pool.map(
                    lambda scale_m: _local_relief_for_scale(
                        elevation, x_resolution, y_resolution, scale_m
                    ),
                    scales_m,
                )
            )
    return {scale_m: relief for scale_m, relief in results}

def _robust_z(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    centre = float(np.median(finite))
    mad = float(np.median(np.abs(finite - centre)))
    scale = max(1e-6, 1.4826 * mad)
    return float((np.mean(finite) - centre) / scale)


def _geometry_orientation_degrees(geometry: Any) -> float:
    """Return the dominant long-axis orientation in degrees [0, 180)."""
    minimum = geometry.minimum_rotated_rectangle
    if minimum.is_empty or not hasattr(minimum, "exterior"):
        return 0.0
    coords = list(minimum.exterior.coords)
    if len(coords) < 2:
        return 0.0
    best_length = -1.0
    best_angle = 0.0
    for left, right in zip(coords, coords[1:]):
        dx = float(right[0] - left[0])
        dy = float(right[1] - left[1])
        length = float(np.hypot(dx, dy))
        if length > best_length:
            best_length = length
            best_angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
    return best_angle


def _orientation_difference_degrees(left: float, right: float) -> float:
    difference = abs(left - right) % 180.0
    return min(difference, 180.0 - difference)


def _near_parallel_known_feature(
    geometry: Any,
    aim_features: list[dict[str, Any]],
    *,
    distance_m: float = 10.0,
    angle_tolerance_degrees: float = 18.0,
) -> bool:
    """Reject elongated candidates which hug a mapped feature in parallel.

    This specifically targets the detector drawing a second line alongside an
    existing bank/ditch outline. Perpendicular crossings are retained so an
    unregistered feature can intersect an already mapped one.
    """
    if geometry.is_empty:
        return False
    metrics = _geometry_metrics(geometry)
    if metrics["elongation"] < 5.0:
        return False
    candidate_orientation = _geometry_orientation_degrees(geometry)
    for feature in aim_features:
        known = _safe_geometry(feature.get("geometry"))
        if known is None or known.is_empty:
            continue
        if float(geometry.distance(known)) > distance_m:
            continue
        known_orientation = _geometry_orientation_degrees(known)
        if _orientation_difference_degrees(candidate_orientation, known_orientation) <= angle_tolerance_degrees:
            return True
    return False


def _geometry_metrics(geometry: Any) -> dict[str, float]:
    area = max(float(geometry.area), 1e-6)
    perimeter = max(float(geometry.length), 1e-6)
    hull_area = max(float(geometry.convex_hull.area), area)
    minimum = geometry.minimum_rotated_rectangle
    coords = list(minimum.exterior.coords) if hasattr(minimum, "exterior") else []
    edge_lengths = []
    for left, right in zip(coords, coords[1:]):
        edge_lengths.append(float(((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2) ** 0.5))
    if len(edge_lengths) >= 4:
        edge_lengths.sort(reverse=True)
        long_edge = max(edge_lengths[0], 1e-6)
        short_edge = max(edge_lengths[-1], 1e-6)
        elongation = long_edge / short_edge
        rectangle_area = max(long_edge * short_edge, area)
    else:
        elongation = 1.0
        rectangle_area = area
    holes = 0
    if geometry.geom_type == "Polygon":
        holes = len(geometry.interiors)
    elif geometry.geom_type == "MultiPolygon":
        holes = sum(len(part.interiors) for part in geometry.geoms)
    circularity = float(min(1.0, max(0.0, 4.0 * pi * area / (perimeter * perimeter))))
    rectangularity = float(min(1.0, max(0.0, area / rectangle_area)))
    return {
        "area_m2": area,
        "perimeter_m": perimeter,
        "elongation": float(min(elongation, 100.0)),
        "circularity": circularity,
        "rectangularity": rectangularity,
        "convexity": float(min(1.0, area / hull_area)),
        "holes": float(holes),
    }


def _morphology_score(metrics: dict[str, float], linear_score: float = 0.0) -> float:
    ring = 1.0 if metrics["holes"] > 0 else max(0.0, min(1.0, (metrics["circularity"] - 0.35) / 0.40))
    linear_shape = max(0.0, min(1.0, (metrics["elongation"] - 2.5) / 10.0))
    compact = 0.55 * metrics["circularity"] + 0.45 * metrics["rectangularity"]
    archaeological_linear = linear_shape * float(np.clip(linear_score, 0.0, 1.0))
    score = 0.25 + 0.25 * ring + 0.25 * compact + 0.25 * archaeological_linear
    return float(np.clip(score, 0.0, 1.0))


def _robust_unit(values: np.ndarray, high_quantile: float = 0.995) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.zeros_like(values, dtype="float32")
    if not finite.any():
        return result
    sample = values[finite]
    centre = float(np.median(sample))
    mad = float(np.median(np.abs(sample - centre)))
    scale = max(1e-6, 1.4826 * mad)
    z = np.abs((values - centre) / scale)
    z[~finite] = 0.0
    high = float(np.quantile(z[finite], high_quantile))
    if high <= 1e-6:
        high = 1.0
    return np.clip(z / high, 0.0, 1.0).astype("float32")


def _linear_response_for_scale(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    scale_m: float,
) -> tuple[float, np.ndarray]:
    from scipy.ndimage import gaussian_filter, sobel

    sigma_x = max(0.8, scale_m / (2.355 * x_resolution))
    sigma_y = max(0.8, scale_m / (2.355 * y_resolution))
    smooth = gaussian_filter(elevation, sigma=(sigma_y, sigma_x), mode="nearest")
    relief = elevation - smooth

    gx = sobel(smooth, axis=1, mode="nearest") / max(8.0 * x_resolution, 1e-6)
    gy = sobel(smooth, axis=0, mode="nearest") / max(8.0 * y_resolution, 1e-6)
    gradient = np.hypot(gx, gy).astype("float32")

    jxx = gaussian_filter(gx * gx, sigma=1.5, mode="nearest")
    jxy = gaussian_filter(gx * gy, sigma=1.5, mode="nearest")
    jyy = gaussian_filter(gy * gy, sigma=1.5, mode="nearest")
    trace = jxx + jyy
    determinant_root = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    coherence = np.divide(
        determinant_root,
        trace + 1e-9,
        out=np.zeros_like(trace, dtype="float32"),
        where=trace > 1e-9,
    )
    coherence = np.clip(coherence, 0.0, 1.0).astype("float32")

    laplacian = np.gradient(np.gradient(smooth, axis=1), axis=1)
    laplacian += np.gradient(np.gradient(smooth, axis=0), axis=0)
    curvature = _robust_unit(np.abs(laplacian).astype("float32"), 0.995)
    relief_strength = _robust_unit(np.abs(relief).astype("float32"), 0.995)
    edge_strength = _robust_unit(gradient, 0.995)

    response = relief_strength * edge_strength * (0.40 + 0.40 * coherence + 0.20 * curvature)
    return scale_m, response.astype("float32")


def calculate_linear_anomaly_response(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    scales_m: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0),
    *,
    workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect persistent linear earthwork/edge signatures across scales."""
    elevation = _fill_invalid(elevation)
    resolved_workers = _resolve_workers(workers, len(scales_m), elevation.size)
    if resolved_workers == 1:
        results = [
            _linear_response_for_scale(elevation, x_resolution, y_resolution, scale_m)
            for scale_m in scales_m
        ]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="linear") as pool:
            results = list(
                pool.map(
                    lambda scale_m: _linear_response_for_scale(
                        elevation, x_resolution, y_resolution, scale_m
                    ),
                    scales_m,
                )
            )

    best_response = np.zeros_like(elevation, dtype="float32")
    best_scale = np.zeros_like(elevation, dtype="float32")
    for scale_m, response in results:
        selector = response > best_response
        best_response[selector] = response[selector]
        best_scale[selector] = scale_m
    return best_response, best_scale

def _straight_line_support_mask(
    response: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    *,
    threshold_percentile: float = 97.25,
    min_line_length_m: float = 20.0,
    dilation_m: float = 2.5,
) -> np.ndarray:
    """Seed persistent, coherent straight-line terrain candidates with Hough lines."""
    import math

    from scipy.ndimage import binary_dilation
    from skimage.draw import line as draw_line
    from skimage.transform import probabilistic_hough_line

    finite = np.isfinite(response)
    if finite.sum() < 100:
        return np.zeros_like(response, dtype=bool)
    values = response[finite]
    threshold = float(np.percentile(values, threshold_percentile))
    edge = (response >= threshold) & finite
    if not edge.any():
        return np.zeros_like(response, dtype=bool)

    height, width = edge.shape
    downsample = max(1, int(math.ceil(max(height, width) / 1400)))
    if downsample > 1:
        from skimage.transform import resize
        small = resize(
            edge.astype("uint8"),
            (
                max(64, int(math.ceil(height / downsample))),
                max(64, int(math.ceil(width / downsample))),
            ),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(bool)
    else:
        small = edge

    pixel_size = max(0.25, min(abs(float(x_resolution)), abs(float(y_resolution))))
    minimum_length_px = max(10, int(round(min_line_length_m / pixel_size / downsample)))
    gap_px = max(3, int(round(10.0 / pixel_size / downsample)))
    theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, 120, endpoint=False)
    lines = probabilistic_hough_line(
        small,
        threshold=max(5, minimum_length_px // 4),
        line_length=minimum_length_px,
        line_gap=gap_px,
        theta=theta,
    )

    support = np.zeros_like(edge, dtype=bool)
    for (x0, y0), (x1, y1) in lines[:500]:
        if downsample > 1:
            x0 = int(round(x0 * downsample))
            y0 = int(round(y0 * downsample))
            x1 = int(round(x1 * downsample))
            y1 = int(round(y1 * downsample))
        length = float(np.hypot(x1 - x0, y1 - y0)) * pixel_size
        if length < min_line_length_m:
            continue
        rr, cc = draw_line(
            int(np.clip(y0, 0, height - 1)),
            int(np.clip(x0, 0, width - 1)),
            int(np.clip(y1, 0, height - 1)),
            int(np.clip(x1, 0, width - 1)),
        )
        support[rr, cc] = True

    dilation_px = max(1, int(round(dilation_m / pixel_size)))
    return binary_dilation(support, structure=np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), dtype=bool))

def _signature(metrics: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            np.log1p(metrics["elongation"]),
            metrics["circularity"],
            metrics["rectangularity"],
            metrics["convexity"],
            min(metrics["holes"], 4.0),
            np.log1p(metrics["perimeter_m"] / max(np.sqrt(metrics["area_m2"]), 1.0)),
        ],
        dtype="float64",
    )


def _he_similarity(candidate_metrics: dict[str, float], aim_features: list[dict[str, Any]]) -> float:
    signatures: list[np.ndarray] = []
    for feature in aim_features:
        geometry_data = feature.get("geometry")
        if not geometry_data:
            continue
        try:
            geometry = shape(geometry_data)
        except Exception:
            continue
        if geometry.is_empty:
            continue
        try:
            signatures.append(_signature(_geometry_metrics(geometry)))
        except Exception:
            continue
    if not signatures:
        return 0.0
    target = _signature(candidate_metrics)
    distances = [float(np.linalg.norm(target - reference)) for reference in signatures]
    distance = min(distances)
    return float(np.exp(-0.70 * distance))


def _candidate_raster_values(geometry: Any, raster: np.ndarray, transform: Any) -> np.ndarray:
    from rasterio.features import rasterize

    mask = rasterize(
        [(geometry, 1)],
        out_shape=raster.shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    values = raster[mask]
    return values[np.isfinite(values)]


def _candidate_raster_score(geometry: Any, raster: np.ndarray, transform: Any) -> float:
    values = _candidate_raster_values(geometry, raster, transform)
    return float(np.mean(values)) if values.size else 0.0


def _candidate_raster_tail_mean(
    geometry: Any, raster: np.ndarray, transform: Any, quantile: float = 0.90
) -> float:
    """Return the mean of the strongest raster response in a candidate."""
    values = _candidate_raster_values(geometry, raster, transform)
    if values.size == 0:
        return 0.0
    cutoff = float(np.quantile(values, quantile))
    strongest = values[values >= cutoff]
    return float(np.mean(strongest)) if strongest.size else cutoff


def _safe_geometry(geometry_data: Any) -> Any | None:
    if not geometry_data:
        return None
    try:
        from shapely.validation import make_valid
        geometry = shape(geometry_data)
        if geometry.is_empty:
            return None
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        if geometry.is_empty:
            return None
        return geometry.buffer(0) if not geometry.is_valid else geometry
    except Exception:
        try:
            return shape(geometry_data).buffer(0)
        except Exception:
            return None


def _geometry_union(features: list[dict[str, Any]], buffer_m: float) -> Any | None:
    geometries: list[Any] = []
    for feature in features:
        geometry = _safe_geometry(feature.get("geometry"))
        if geometry is None or geometry.is_empty:
            continue
        try:
            geometries.append(geometry.buffer(buffer_m) if buffer_m > 0 else geometry)
        except Exception:
            continue
    if not geometries:
        return None
    try:
        return unary_union(geometries)
    except Exception:
        return None


def _rasterise_geometry_mask(
    geometry: Any | None,
    shape_out: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    from rasterio.features import rasterize
    if geometry is None or geometry.is_empty:
        return np.zeros(shape_out, dtype=bool)
    return rasterize(
        [(geometry, 1)],
        out_shape=shape_out,
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)


def _sensitivity_profile(name: str) -> SensitivityProfile:
    try:
        return SENSITIVITY_PROFILES[name.casefold()]
    except KeyError as exc:
        valid = ", ".join(sorted(SENSITIVITY_PROFILES))
        raise ValueError(f"Unknown anomaly sensitivity '{name}'. Choose one of: {valid}.") from exc


def detect_terrain_anomalies(
    dtm_path: Path,
    aim_features: list[dict[str, Any]],
    *,
    monument_extents: list[dict[str, Any]] | None = None,
    modern_features: list[dict[str, Any]] | None = None,
    modern_context_path: Path | None = None,
    satellite_support_path: Path | None = None,
    sensitivity: str = "medium",
    workers: int = 0,
    known_buffer_m: float | None = None,
    threshold_percentile: float | None = None,
    min_area_m2: float | None = None,
    max_area_m2: float | None = None,
    max_candidates: int | None = None,
) -> tuple[list[TerrainCandidate], dict[str, Any]]:
    """Generate independent LiDAR anomalies and rank them with contextual evidence.

    Historic England features are context and exclusion masks only; an empty HE
    dataset is a valid input and produces candidates from terrain alone.
    """
    import rasterio
    from rasterio.features import shapes
    from scipy.ndimage import binary_closing, label
    from prospector.terrain.context import build_modern_context_raster

    profile = _sensitivity_profile(sensitivity)
    profile = SensitivityProfile(
        name=profile.name,
        relief_scales_m=profile.relief_scales_m,
        threshold_percentile=profile.threshold_percentile if threshold_percentile is None else threshold_percentile,
        linear_percentile=profile.linear_percentile,
        hough_percentile=profile.hough_percentile,
        min_line_length_m=profile.min_line_length_m,
        min_area_m2=profile.min_area_m2 if min_area_m2 is None else min_area_m2,
        max_area_m2=profile.max_area_m2 if max_area_m2 is None else max_area_m2,
        max_candidates=profile.max_candidates if max_candidates is None else max_candidates,
        detailed_buffer_m=profile.detailed_buffer_m if known_buffer_m is None else known_buffer_m,
        monument_buffer_m=profile.monument_buffer_m,
        closing_pixels=profile.closing_pixels,
    )
    monument_extents = monument_extents or []
    modern_features = modern_features or []

    with rasterio.open(dtm_path) as dataset:
        data = dataset.read(1, masked=True).filled(np.nan).astype("float32")
        transform = dataset.transform
        x_resolution, y_resolution = dataset.res
        shape_out = data.shape

    filled = _fill_invalid(data)
    reliefs = calculate_local_relief_models(
        filled,
        x_resolution,
        y_resolution,
        profile.relief_scales_m,
        workers=workers,
    )
    z_scores: dict[float, np.ndarray] = {}
    for scale_m, relief in reliefs.items():
        centre = float(np.median(relief))
        mad = float(np.median(np.abs(relief - centre)))
        scale = max(1e-6, 1.4826 * mad)
        z_scores[scale_m] = np.abs((relief - centre) / scale)

    score = np.max(np.stack(list(z_scores.values()), axis=0), axis=0)
    linear_response, _linear_scale = calculate_linear_anomaly_response(
        filled,
        x_resolution,
        y_resolution,
        tuple(scale for scale in (4.0, 8.0, 16.0, 32.0) if scale <= max(profile.relief_scales_m)),
        workers=workers,
    )

    strongest_scale = np.zeros(shape_out, dtype="float32")
    strongest_relief = np.zeros(shape_out, dtype="float32")
    best_z = np.full(shape_out, -np.inf, dtype="float32")
    for scale_m, z_score in z_scores.items():
        selector = z_score > best_z
        strongest_scale[selector] = scale_m
        strongest_relief[selector] = reliefs[scale_m][selector]
        best_z[selector] = z_score[selector]

    valid = np.isfinite(data)
    valid_scores = score[valid]
    if valid_scores.size == 0:
        return [], {"name": "multi-scale-local-relief", "status": "no-data"}

    threshold = float(np.percentile(valid_scores, profile.threshold_percentile))
    anomaly_mask = (score >= threshold) & valid
    linear_values = linear_response[valid]
    linear_threshold = float(np.percentile(linear_values, profile.linear_percentile)) if linear_values.size else 1.0
    linear_mask = (linear_response >= linear_threshold) & valid
    straight_line_mask = _straight_line_support_mask(
        linear_response,
        x_resolution,
        y_resolution,
        threshold_percentile=profile.hough_percentile,
        min_line_length_m=profile.min_line_length_m,
    ) & valid

    detail_union = _geometry_union(aim_features, profile.detailed_buffer_m)
    monument_union = _geometry_union(monument_extents, profile.monument_buffer_m)
    detail_mask = _rasterise_geometry_mask(detail_union, shape_out, transform)
    monument_mask = _rasterise_geometry_mask(monument_union, shape_out, transform)
    exclusion_mask = detail_mask | monument_mask

    combined_mask = (anomaly_mask | linear_mask | straight_line_mask) & ~exclusion_mask
    structure_size = max(3, profile.closing_pixels * 2 + 1)
    combined_mask = binary_closing(
        combined_mask,
        structure=np.ones((structure_size, structure_size), dtype=bool),
    )
    combined_mask &= ~exclusion_mask
    combined_mask &= valid
    labels, _count = label(combined_mask, structure=np.ones((3, 3), dtype=np.uint8))

    modern_score = np.zeros(shape_out, dtype="float32")
    modern_metadata: dict[str, Any] = {"enabled": False}
    if modern_context_path is not None and modern_context_path.is_file():
        with rasterio.open(modern_context_path) as modern_raster:
            modern_score = modern_raster.read(1).astype("float32")
        modern_metadata = {"enabled": True, "source": str(modern_context_path)}
    elif modern_features:
        temporary = dtm_path.parent / ".modern-context-score.tif"
        modern = build_modern_context_raster(dtm_path, modern_features, temporary)
        with rasterio.open(modern.score_path) as modern_raster:
            modern_score = modern_raster.read(1).astype("float32")
        modern_metadata = {"enabled": True, **modern.metadata}
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(temporary.suffix + ".json").unlink(missing_ok=True)

    satellite_score = np.zeros(shape_out, dtype="float32")
    satellite_metadata: dict[str, Any] = {"enabled": False}
    if satellite_support_path is not None and satellite_support_path.is_file():
        with rasterio.open(satellite_support_path) as sat:
            satellite_score = np.nan_to_num(sat.read(1).astype("float32"), nan=0.0)
        satellite_metadata = {"enabled": True, "source": str(satellite_support_path)}

    pixel_area = abs(float(x_resolution) * float(y_resolution))
    candidate_geometries: dict[int, list[Any]] = {}
    for geometry_data, value in shapes(labels.astype("int32"), mask=labels > 0, transform=transform):
        candidate_geometries.setdefault(int(value), []).append(shape(geometry_data))

    provisional: list[TerrainCandidate] = []
    provisional_stats: dict[str, int] = {
        "components_considered": 0,
        "modern_heavily_penalised": 0,
        "known_outline_traces_rejected": 0,
        "kept_for_ranking": 0,
    }
    next_id = 1
    for label_id, geometries in candidate_geometries.items():
        provisional_stats["components_considered"] += 1
        pixels = labels == label_id
        pixel_count = int(pixels.sum())
        component_area = pixel_count * pixel_area
        if component_area < profile.min_area_m2 or component_area > profile.max_area_m2:
            continue
        geometry = unary_union(geometries).buffer(0)
        if detail_union is not None:
            geometry = geometry.difference(detail_union).buffer(0)
        if monument_union is not None:
            geometry = geometry.difference(monument_union).buffer(0)
        if geometry.is_empty:
            continue
        actual_area = float(geometry.area)
        if actual_area < profile.min_area_m2:
            continue
        if _near_parallel_known_feature(geometry, aim_features):
            provisional_stats["known_outline_traces_rejected"] += 1
            continue

        strongest_values = np.abs(strongest_relief[pixels])
        relief_m = float(np.max(strongest_values)) if strongest_values.size else 0.0
        strongest_index = int(np.argmax(strongest_values)) if strongest_values.size else 0
        raw_relief = float(strongest_relief[pixels][strongest_index]) if strongest_values.size else 0.0
        polarity = "positive" if raw_relief >= 0 else "negative"
        scale_m = float(strongest_scale[pixels][strongest_index]) if strongest_values.size else 0.0

        morphology = _geometry_metrics(geometry)
        linear_score = float(np.clip(_candidate_raster_tail_mean(geometry, linear_response, transform, 0.75), 0.0, 1.0))
        shape_score = _morphology_score(morphology, linear_score)
        persistence_values = np.stack(
            [(z_scores[scale_m][pixels] >= 2.0).astype("float32") for scale_m in sorted(z_scores)],
            axis=0,
        )
        persistence = float(np.mean(persistence_values))
        lidar_values = score[pixels]
        lidar_tail = float(np.quantile(lidar_values, 0.85)) if lidar_values.size else 0.0
        lidar_strength = float(np.clip(lidar_tail / 5.0, 0.0, 1.0))
        modern_penalty = float(np.clip(_candidate_raster_tail_mean(geometry, modern_score, transform), 0.0, 1.0))
        satellite_support = float(np.clip(_candidate_raster_score(geometry, satellite_score, transform), 0.0, 1.0))
        he_similarity = _he_similarity(morphology, aim_features)

        # LiDAR is the discovery signal. HE similarity is deliberately tiny and
        # cannot manufacture a candidate; it is only weak contextual evidence.
        linear_bonus = 0.05 if linear_score >= 0.65 and morphology["elongation"] >= 8.0 else 0.0
        base = (
            0.34 * lidar_strength
            + 0.18 * persistence
            + 0.13 * shape_score
            + 0.25 * linear_score
            + 0.04 * satellite_support
            + 0.02 * he_similarity
            + linear_bonus
        )

        # Modernity is contextual, not a veto. Strongly modern geometry is
        # down-ranked, but a very linear/archaeological-looking candidate survives.
        modern_multiplier = 1.0 - 0.78 * modern_penalty
        final_score = 100.0 * max(0.0, base * modern_multiplier)

        reasons: list[str] = []
        if shape_score >= 0.70:
            reasons.append("coherent archaeological morphology")
        if morphology["holes"] > 0:
            reasons.append("closed or ring-like geometry")
        if persistence >= 0.50:
            reasons.append("persists across multiple LiDAR scales")
        if satellite_support >= 0.20:
            reasons.append("satellite vegetation/reflectance anomaly supports the terrain signal")
        if he_similarity >= 0.75:
            reasons.append("geometry resembles known Historic England mapped forms")
        if modern_penalty >= 0.65:
            reasons.append("strong modern-feature conflict")
            provisional_stats["modern_heavily_penalised"] += 1
        if linear_score >= 0.55:
            reasons.append("strong coherent linear LiDAR response")
        if morphology["elongation"] >= 10.0:
            reasons.append("very elongated linear morphology")
        if linear_bonus:
            reasons.append("persistent straight-line support across the AOI")
        provisional_stats["kept_for_ranking"] += 1

        provisional.append(
            TerrainCandidate(
                candidate_id=next_id,
                geometry=geometry,
                score=final_score,
                area_m2=actual_area,
                relief_m=relief_m,
                polarity=polarity,
                strongest_scale_m=scale_m,
                lidar_score=100.0 * lidar_strength,
                persistence_score=100.0 * persistence,
                morphology_score=100.0 * shape_score,
                modern_penalty=100.0 * modern_penalty,
                satellite_support=100.0 * satellite_support,
                he_similarity=100.0 * he_similarity,
                classification="candidate",
                reasons=tuple(reasons),
                morphology=tuple(sorted((key, float(value)) for key, value in morphology.items())),
                linear_score=100.0 * linear_score,
            )
        )
        next_id += 1

    provisional.sort(key=lambda item: (item.score, item.linear_score, item.persistence_score, item.relief_m), reverse=True)
    retained: list[TerrainCandidate] = []
    rejected_modern = 0
    for candidate in provisional:
        morphology = dict(candidate.morphology)
        if candidate.modern_penalty >= 97.0 and morphology.get("elongation", 0.0) < 6.0 and candidate.morphology_score < 60.0:
            rejected_modern += 1
            continue
        retained.append(candidate)
        if len(retained) >= profile.max_candidates:
            break

    retained = [
        TerrainCandidate(
            candidate_id=index,
            geometry=item.geometry,
            score=item.score,
            area_m2=item.area_m2,
            relief_m=item.relief_m,
            polarity=item.polarity,
            strongest_scale_m=item.strongest_scale_m,
            lidar_score=item.lidar_score,
            persistence_score=item.persistence_score,
            morphology_score=item.morphology_score,
            modern_penalty=item.modern_penalty,
            satellite_support=item.satellite_support,
            he_similarity=item.he_similarity,
            classification="high-priority" if item.score >= 70.0 else "candidate",
            reasons=item.reasons,
            morphology=item.morphology,
            linear_score=item.linear_score,
        )
        for index, item in enumerate(retained, 1)
    ]

    metadata = {
        "name": "multi-scale-local-relief",
        "version": "0.3.7",
        "sensitivity": profile.name,
        "workers_requested": workers,
        "workers_auto_cap": 4,
        "historic_england_reference": {
            "detail": "Historic England is contextual evidence and an exclusion mask; candidate generation remains valid with zero HE records.",
            "detailed_mapping_count": len(aim_features),
            "monument_extents_count": len(monument_extents),
            "monument_extents_exclusion": bool(monument_extents),
            "monument_extents_buffer_m": profile.monument_buffer_m,
            "detailed_mapping_exclusion_buffer_m": profile.detailed_buffer_m,
            "candidate_generation_requires_historic_england": False,
        },
        "threshold_percentile": profile.threshold_percentile,
        "linear_response_percentile": profile.linear_percentile,
        "hough_seed_percentile": profile.hough_percentile,
        "straight_line_seeded": True,
        "minimum_hough_line_length_m": profile.min_line_length_m,
        "threshold_z": threshold,
        "scales_m": list(reliefs),
        "cross_scale_persistence_threshold_z": 2.0,
        "min_area_m2": profile.min_area_m2,
        "max_area_m2": profile.max_area_m2,
        "max_candidates": profile.max_candidates,
        "provisional_candidate_count": len(provisional),
        "candidate_count": len(retained),
        "rejected_for_modern_context": rejected_modern,
        "modern_context": modern_metadata,
        "satellite_context": satellite_metadata,
        "ranking": {
            "lidar": 0.34,
            "persistence": 0.18,
            "morphology": 0.13,
            "linear_response": 0.25,
            "satellite": 0.04,
            "historic_england_similarity": 0.02,
            "linear_structure_bonus": 0.05,
            "modern_penalty_strength": 0.78,
        },
        "provisional_statistics": provisional_stats,
        "warning": (
            "Candidates are ranked by deterministic LiDAR/context heuristics. "
            "They are research leads, not confirmed archaeological identifications."
        ),
    }
    return retained, metadata

