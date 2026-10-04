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
    terrain_novelty_score: float = 0.0
    ring_score: float = 0.0
    ridge_valley_score: float = 0.0
    texture_score: float = 0.0


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
    level: int
    threshold_percentile: float
    seed_percentile: float
    linear_percentile: float
    novelty_percentile: float
    annular_percentile: float
    min_line_length_m: float
    min_area_m2: float
    max_area_m2: float
    max_candidates: int
    detailed_buffer_m: float
    monument_buffer_m: float
    closing_pixels: int
    relief_scales_m: tuple[float, ...]
    pattern_scales_m: tuple[float, ...]
    ml_sample_size: int


def _profile_for_level(level: int) -> SensitivityProfile:
    """Create a smooth 1-10 recall/selectivity curve."""
    level = max(1, min(10, int(level)))
    return SensitivityProfile(
        level=level,
        threshold_percentile=99.70 - (level - 1) * 0.58,
        seed_percentile=max(90.0, 99.25 - (level - 1) * 0.58),
        linear_percentile=99.25 - (level - 1) * 0.62,
        novelty_percentile=99.30 - (level - 1) * 0.68,
        annular_percentile=99.35 - (level - 1) * 0.67,
        min_line_length_m=max(7.0, 36.0 - level * 2.6),
        min_area_m2=max(1.5, 22.0 - (level - 1) * 2.5),
        max_area_m2=min(30000.0, 10000.0 + level * 1800.0),
        max_candidates=35 + level * 70,
        detailed_buffer_m=max(2.5, 5.0 - level * 0.18),
        monument_buffer_m=2.0,
        closing_pixels=1 if level >= 8 else 2,
        relief_scales_m=tuple(
            scale for scale in (1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 24.0, 36.0, 54.0, 80.0)
            if scale <= (10.0 + level * 8.0)
        ),
        pattern_scales_m=tuple(
            scale for scale in (3.0, 6.0, 12.0, 24.0, 48.0)
            if scale <= (12.0 + level * 5.0)
        ),
        ml_sample_size=12000 + level * 2500,
    )


SENSITIVITY_PROFILES: dict[str, SensitivityProfile] = {
    str(level): _profile_for_level(level) for level in range(1, 11)
}
# Backwards-compatible aliases for V0.3 command lines.
SENSITIVITY_PROFILES.update({
    "low": SENSITIVITY_PROFILES["2"],
    "medium": SENSITIVITY_PROFILES["5"],
    "high": SENSITIVITY_PROFILES["8"],
})


def _resolve_workers(requested: int, task_count: int, pixels: int) -> int:
    if requested > 0:
        return max(1, min(requested, task_count))
    if task_count <= 1 or pixels < 250_000:
        return 1
    import os
    return max(1, min(task_count, min(6, (os.cpu_count() or 2))))


def _local_relief_for_scale(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    scale_m: float,
) -> tuple[float, np.ndarray]:
    from scipy.ndimage import gaussian_filter

    sigma_x = max(0.7, scale_m / (2.355 * abs(x_resolution)))
    sigma_y = max(0.7, scale_m / (2.355 * abs(y_resolution)))
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
            results = list(pool.map(
                lambda scale_m: _local_relief_for_scale(elevation, x_resolution, y_resolution, scale_m),
                scales_m,
            ))
    return {scale_m: relief for scale_m, relief in results}


def _geometry_orientation_degrees(geometry: Any) -> float:
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
    edge_lengths = [
        float(np.hypot(right[0] - left[0], right[1] - left[1]))
        for left, right in zip(coords, coords[1:])
    ]
    if len(edge_lengths) >= 4:
        edge_lengths.sort(reverse=True)
        long_edge = max(edge_lengths[0], 1e-6)
        short_edge = max(edge_lengths[-1], 1e-6)
        elongation = long_edge / short_edge
        rectangle_area = max(long_edge * short_edge, area)
    else:
        elongation = 1.0
        rectangle_area = area
    if geometry.geom_type == "Polygon":
        holes = len(geometry.interiors)
    elif geometry.geom_type == "MultiPolygon":
        holes = sum(len(part.interiors) for part in geometry.geoms)
    else:
        holes = 0
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
    ring = 1.0 if metrics["holes"] > 0 else max(0.0, min(1.0, (metrics["circularity"] - 0.28) / 0.52))
    linear_shape = max(0.0, min(1.0, (metrics["elongation"] - 2.5) / 10.0))
    compact = 0.55 * metrics["circularity"] + 0.45 * metrics["rectangularity"]
    archaeological_linear = linear_shape * float(np.clip(linear_score, 0.0, 1.0))
    return float(np.clip(0.20 + 0.27 * ring + 0.25 * compact + 0.28 * archaeological_linear, 0.0, 1.0))


def _robust_unit(values: np.ndarray, high_quantile: float = 0.995) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.zeros_like(values, dtype="float32")
    if not finite.any():
        return result
    sample = values[finite].astype("float32", copy=False)
    centre = float(np.median(sample))
    mad = float(np.median(np.abs(sample - centre)))
    scale = max(1e-6, 1.4826 * mad)
    z = np.abs((values - centre) / scale)
    z[~finite] = 0.0
    high = float(np.quantile(z[finite], high_quantile))
    high = max(high, 1.0)
    return np.clip(z / high, 0.0, 1.0).astype("float32")


def _linear_response_for_scale(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    scale_m: float,
) -> tuple[float, np.ndarray]:
    from scipy.ndimage import gaussian_filter

    sigma_x = max(0.7, scale_m / (2.355 * abs(x_resolution)))
    sigma_y = max(0.7, scale_m / (2.355 * abs(y_resolution)))
    smooth = gaussian_filter(elevation, sigma=(sigma_y, sigma_x), mode="nearest")
    relief = elevation - smooth
    gy, gx = np.gradient(smooth, y_resolution, x_resolution)
    gradient = np.hypot(gx, gy).astype("float32")

    jxx = gaussian_filter(gx * gx, sigma=1.4, mode="nearest")
    jxy = gaussian_filter(gx * gy, sigma=1.4, mode="nearest")
    jyy = gaussian_filter(gy * gy, sigma=1.4, mode="nearest")
    trace = jxx + jyy
    determinant_root = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    coherence = np.divide(determinant_root, trace + 1e-9, out=np.zeros_like(trace), where=trace > 1e-9)
    curvature = _robust_unit(np.abs(np.gradient(gx, axis=1) / max(abs(x_resolution), 1e-6) + np.gradient(gy, axis=0) / max(abs(y_resolution), 1e-6)).astype("float32"), 0.997)
    relief_strength = _robust_unit(np.abs(relief).astype("float32"), 0.997)
    edge_strength = _robust_unit(gradient, 0.997)
    response = relief_strength * edge_strength * (0.30 + 0.50 * np.clip(coherence, 0.0, 1.0) + 0.20 * curvature)
    return scale_m, response.astype("float32")


def calculate_linear_anomaly_response(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    scales_m: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0),
    *,
    workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    elevation = _fill_invalid(elevation)
    resolved_workers = _resolve_workers(workers, len(scales_m), elevation.size)
    if resolved_workers == 1:
        results = [_linear_response_for_scale(elevation, x_resolution, y_resolution, scale) for scale in scales_m]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=resolved_workers, thread_name_prefix="linear") as pool:
            results = list(pool.map(lambda scale: _linear_response_for_scale(elevation, x_resolution, y_resolution, scale), scales_m))
    best = np.zeros_like(elevation, dtype="float32")
    best_scale = np.zeros_like(elevation, dtype="float32")
    for scale, response in results:
        selector = response > best
        best[selector] = response[selector]
        best_scale[selector] = scale
    return best, best_scale


def _straight_line_support_mask(
    response: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    *,
    threshold_percentile: float = 97.25,
    min_line_length_m: float = 20.0,
    dilation_m: float = 2.5,
) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    from skimage.draw import line as draw_line
    from skimage.transform import probabilistic_hough_line
    finite = np.isfinite(response)
    if finite.sum() < 100:
        return np.zeros_like(response, dtype=bool)
    threshold = float(np.percentile(response[finite], threshold_percentile))
    edge = (response >= threshold) & finite
    if not edge.any():
        return np.zeros_like(response, dtype=bool)
    height, width = edge.shape
    pixel_size = max(0.25, min(abs(float(x_resolution)), abs(float(y_resolution))))
    lines = probabilistic_hough_line(
        edge,
        threshold=max(4, int(round(min_line_length_m / pixel_size / 5.0))),
        line_length=max(8, int(round(min_line_length_m / pixel_size))),
        line_gap=max(2, int(round(10.0 / pixel_size))),
        theta=np.linspace(-np.pi / 2.0, np.pi / 2.0, 90, endpoint=False),
        rng=42,
    )
    support = np.zeros_like(edge, dtype=bool)
    for (x0, y0), (x1, y1) in lines[:800]:
        length = float(np.hypot(x1 - x0, y1 - y0)) * pixel_size
        if length < min_line_length_m:
            continue
        rr, cc = draw_line(int(np.clip(y0, 0, height - 1)), int(np.clip(x0, 0, width - 1)), int(np.clip(y1, 0, height - 1)), int(np.clip(x1, 0, width - 1)))
        support[rr, cc] = True
    dilation_px = max(1, int(round(dilation_m / pixel_size)))
    return binary_dilation(support, structure=np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), dtype=bool))


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
    geometries = []
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


def _rasterise_geometry_mask(geometry: Any | None, shape_out: tuple[int, int], transform: Any) -> np.ndarray:
    from rasterio.features import rasterize
    if geometry is None or geometry.is_empty:
        return np.zeros(shape_out, dtype=bool)
    return rasterize([(geometry, 1)], out_shape=shape_out, transform=transform, fill=0, default_value=1, dtype="uint8", all_touched=True).astype(bool)


def _signature(metrics: dict[str, float]) -> np.ndarray:
    return np.asarray([
        np.log1p(metrics["elongation"]), metrics["circularity"], metrics["rectangularity"],
        metrics["convexity"], min(metrics["holes"], 4.0),
        np.log1p(metrics["perimeter_m"] / max(np.sqrt(metrics["area_m2"]), 1.0)),
    ], dtype="float64")


def _he_signatures(aim_features: list[dict[str, Any]]) -> list[np.ndarray]:
    signatures = []
    for feature in aim_features:
        geometry = _safe_geometry(feature.get("geometry"))
        if geometry is None or geometry.is_empty:
            continue
        try:
            signatures.append(_signature(_geometry_metrics(geometry)))
        except Exception:
            continue
    return signatures


def _he_similarity(candidate_metrics: dict[str, float], signatures: list[np.ndarray]) -> float:
    if not signatures:
        return 0.0
    target = _signature(candidate_metrics)
    distance = min(float(np.linalg.norm(target - reference)) for reference in signatures)
    return float(np.exp(-0.70 * distance))


def _candidate_raster_values(geometry: Any, raster: np.ndarray, transform: Any) -> np.ndarray:
    from rasterio.features import rasterize
    mask = rasterize([(geometry, 1)], out_shape=raster.shape, transform=transform, fill=0, default_value=1, dtype="uint8", all_touched=True).astype(bool)
    values = raster[mask]
    return values[np.isfinite(values)]


def _candidate_raster_score(geometry: Any, raster: np.ndarray, transform: Any) -> float:
    values = _candidate_raster_values(geometry, raster, transform)
    return float(np.mean(values)) if values.size else 0.0


def _candidate_raster_tail_mean(geometry: Any, raster: np.ndarray, transform: Any, quantile: float = 0.90) -> float:
    values = _candidate_raster_values(geometry, raster, transform)
    if values.size == 0:
        return 0.0
    cutoff = float(np.quantile(values, quantile))
    strongest = values[values >= cutoff]
    return float(np.mean(strongest)) if strongest.size else cutoff


def _raster_tail_mean_values(raster: np.ndarray, pixels: np.ndarray, quantile: float = 0.90) -> float:
    values = raster[pixels]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    cutoff = float(np.quantile(values, quantile))
    strongest = values[values >= cutoff]
    return float(np.mean(strongest)) if strongest.size else cutoff


def _write_diagnostic_raster(path: Path, data: np.ndarray, reference_path: Path) -> Path:
    """Write a floating-point detector channel using the DTM grid and CRS."""
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0).astype("float32")
    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()
        profile.update(dtype="float32", count=1, nodata=0.0, compress="deflate")
        # A non-tiled reference may still carry block-size hints. Remove them
        # before creating the small diagnostic raster so GDAL does not emit
        # BLOCKXSIZE/BLOCKYSIZE warnings.
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        with rasterio.open(path, "w", **profile) as destination:
            destination.write(values, 1)
    return path


def _resolution_pixels(metres: float, x_resolution: float, y_resolution: float, *, minimum: int = 3) -> int:
    pixels = int(round(metres / max(abs(x_resolution), abs(y_resolution))))
    return max(minimum, pixels | 1)


def _annular_response(elevation: np.ndarray, x_resolution: float, y_resolution: float, scales_m: tuple[float, ...]) -> np.ndarray:
    """Approximate raised/lowered rings with nested local means."""
    from scipy.ndimage import uniform_filter
    best = np.zeros_like(elevation, dtype="float32")
    for scale_m in scales_m:
        outer_size = _resolution_pixels(scale_m * 2.4, x_resolution, y_resolution, minimum=5)
        inner_size = _resolution_pixels(scale_m * 0.75, x_resolution, y_resolution, minimum=3)
        outer = uniform_filter(elevation, size=outer_size, mode="nearest")
        inner = uniform_filter(elevation, size=inner_size, mode="nearest")
        response = _robust_unit(np.abs(outer - inner).astype("float32"), 0.997)
        best = np.maximum(best, response)
    return best


def _ridge_valley_response(elevation: np.ndarray, x_resolution: float, y_resolution: float, scales_m: tuple[float, ...]) -> np.ndarray:
    from scipy.ndimage import maximum_filter, minimum_filter
    best = np.zeros_like(elevation, dtype="float32")
    for scale_m in scales_m:
        size_y = max(3, int(round(scale_m / max(abs(y_resolution), 1e-6))) | 1)
        size_x = max(3, int(round(scale_m / max(abs(x_resolution), 1e-6))) | 1)
        local_max = maximum_filter(elevation, size=(size_y, size_x), mode="nearest")
        local_min = minimum_filter(elevation, size=(size_y, size_x), mode="nearest")
        response = np.maximum(
            _robust_unit((elevation - local_min).astype("float32"), 0.997),
            _robust_unit((local_max - elevation).astype("float32"), 0.997),
        )
        best = np.maximum(best, response)
    return best


def _surface_pattern_maps(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    reliefs: dict[float, np.ndarray],
    linear_response: np.ndarray,
    pattern_scales: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter
    base_scale = min(pattern_scales, key=lambda value: abs(value - 8.0)) if pattern_scales else 8.0
    smooth = gaussian_filter(elevation, sigma=(max(0.7, base_scale / (2.355 * abs(y_resolution))), max(0.7, base_scale / (2.355 * abs(x_resolution)))), mode="nearest")
    gy, gx = np.gradient(smooth, y_resolution, x_resolution)
    slope = np.hypot(gx, gy).astype("float32")
    gx_s = np.gradient(smooth, axis=1) / max(abs(x_resolution), 1e-6)
    gy_s = np.gradient(smooth, axis=0) / max(abs(y_resolution), 1e-6)
    texture = np.zeros_like(elevation, dtype="float32")
    for relief in reliefs.values():
        local = relief - gaussian_filter(relief, sigma=1.2, mode="nearest")
        texture = np.maximum(texture, np.abs(local))
    curvature = np.abs(np.gradient(gx_s, axis=1) / max(abs(x_resolution), 1e-6) + np.gradient(gy_s, axis=0) / max(abs(y_resolution), 1e-6)).astype("float32")
    jxx = gaussian_filter(gx_s * gx_s, sigma=1.6, mode="nearest")
    jxy = gaussian_filter(gx_s * gy_s, sigma=1.6, mode="nearest")
    jyy = gaussian_filter(gy_s * gy_s, sigma=1.6, mode="nearest")
    trace = jxx + jyy
    numerator = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    coherence = np.divide(numerator, trace + 1e-9, out=np.zeros_like(trace), where=trace > 1e-9)
    ridge_valley = _ridge_valley_response(elevation, x_resolution, y_resolution, pattern_scales)
    annular = _annular_response(elevation, x_resolution, y_resolution, pattern_scales)
    texture_score = np.maximum(_robust_unit(texture, 0.997), 0.55 * _robust_unit(curvature, 0.997) + 0.45 * _robust_unit(slope, 0.997) * np.clip(coherence, 0.0, 1.0))
    return annular.astype("float32"), ridge_valley.astype("float32"), texture_score.astype("float32"), np.clip(coherence, 0.0, 1.0).astype("float32")


def _terrain_novelty_map(
    features: np.ndarray,
    valid: np.ndarray,
    *,
    raster_shape: tuple[int, int],
    sample_size: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit an unsupervised per-AOI terrain-pattern novelty model.

    Training uses the available valid terrain population, while inference is
    performed on a spatially reduced grid and interpolated back to the DTM.
    This prevents a million-pixel AOI becoming a million * 100 tree walks.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for the v0.4 terrain novelty detector") from exc

    height, width = raster_shape
    feature_grid = features.reshape(height, width, -1)
    valid_grid = valid.reshape(height, width)
    valid_indices = np.flatnonzero(valid_grid)
    novelty = np.zeros(valid_grid.shape, dtype="float32")
    if valid_indices.size < 256:
        return novelty.reshape(-1), {"enabled": False, "reason": "too_few_valid_pixels", "spatial_step": 1}

    sample_count = min(int(sample_size), valid_indices.size)
    rng = np.random.default_rng(20261004)
    sample_indices = rng.choice(valid_indices, size=sample_count, replace=False)

    # Keep inference below roughly 80k pixels for a responsive 1 km analysis.
    target_inference_pixels = 80_000
    spatial_step = max(1, int(np.ceil(np.sqrt((height * width) / target_inference_pixels))))
    coarse_features = feature_grid[::spatial_step, ::spatial_step].reshape(-1, feature_grid.shape[-1])
    coarse_valid = valid_grid[::spatial_step, ::spatial_step].reshape(-1)
    coarse_valid_indices = np.flatnonzero(coarse_valid)
    if coarse_valid_indices.size < 256:
        spatial_step = 1
        coarse_features = feature_grid.reshape(-1, feature_grid.shape[-1])
        coarse_valid = valid_grid.reshape(-1)
        coarse_valid_indices = np.flatnonzero(coarse_valid)

    model = IsolationForest(
        n_estimators=100,
        max_samples=min(2048, sample_count),
        contamination="auto",
        random_state=20261004,
        n_jobs=max(1, workers) if workers else -1,
    )
    model.fit(features[sample_indices])
    coarse_scores = (-model.decision_function(coarse_features[coarse_valid_indices])).astype("float32")
    lo, hi = np.quantile(coarse_scores, (0.01, 0.99))
    coarse_scores = np.clip(
        (coarse_scores - float(lo)) / max(float(hi - lo), 1e-6),
        0.0,
        1.0,
    )
    coarse = np.zeros(coarse_valid.shape, dtype="float32")
    coarse[coarse_valid_indices] = coarse_scores
    coarse = coarse.reshape(
        (height + spatial_step - 1) // spatial_step,
        (width + spatial_step - 1) // spatial_step,
    )

    from scipy.ndimage import zoom

    zoom_y = height / coarse.shape[0]
    zoom_x = width / coarse.shape[1]
    novelty = zoom(coarse, (zoom_y, zoom_x), order=1, mode="nearest", prefilter=False).astype("float32")
    novelty = novelty[:height, :width]
    novelty[~valid_grid] = 0.0
    return novelty.reshape(-1), {
        "enabled": True,
        "model": "IsolationForest",
        "training_scope": "current AOI only; unsupervised terrain novelty, not archaeological labels",
        "sample_size": int(sample_count),
        "n_estimators": 100,
        "max_samples": min(2048, sample_count),
        "inference_pixels": int(coarse_valid_indices.size),
        "spatial_step": int(spatial_step),
        "random_seed": 20261004,
    }


def _sensitivity_profile(name: str | int) -> SensitivityProfile:
    raw = str(name).casefold().strip()
    aliases = {"low": 2, "medium": 5, "high": 8}
    try:
        level = aliases[raw] if raw in aliases else int(raw)
    except ValueError as exc:
        raise ValueError("Sensitivity must be an integer from 1 to 10 (low/medium/high aliases are also accepted).") from exc
    if not 1 <= level <= 10:
        raise ValueError("Sensitivity must be between 1 and 10.")
    return SENSITIVITY_PROFILES[str(level)]


def detect_terrain_anomalies(
    dtm_path: Path,
    aim_features: list[dict[str, Any]],
    *,
    monument_extents: list[dict[str, Any]] | None = None,
    modern_features: list[dict[str, Any]] | None = None,
    modern_context_path: Path | None = None,
    satellite_support_path: Path | None = None,
    sensitivity: str | int = "5",
    workers: int = 0,
    known_buffer_m: float | None = None,
    threshold_percentile: float | None = None,
    min_area_m2: float | None = None,
    max_area_m2: float | None = None,
    max_candidates: int | None = None,
    diagnostic_raster_dir: Path | None = None,
) -> tuple[list[TerrainCandidate], dict[str, Any]]:
    """Discover and rank anomalies with deterministic morphology + ML novelty."""
    import rasterio
    from rasterio.features import shapes
    from scipy.ndimage import binary_closing, find_objects, label
    from prospector.terrain.context import build_modern_context_raster

    profile = _sensitivity_profile(sensitivity)
    profile = SensitivityProfile(
        level=profile.level,
        threshold_percentile=profile.threshold_percentile if threshold_percentile is None else threshold_percentile,
        seed_percentile=profile.seed_percentile,
        linear_percentile=profile.linear_percentile,
        novelty_percentile=profile.novelty_percentile,
        annular_percentile=profile.annular_percentile,
        min_line_length_m=profile.min_line_length_m,
        min_area_m2=profile.min_area_m2 if min_area_m2 is None else min_area_m2,
        max_area_m2=profile.max_area_m2 if max_area_m2 is None else max_area_m2,
        max_candidates=profile.max_candidates if max_candidates is None else max_candidates,
        detailed_buffer_m=max(1.5, profile.detailed_buffer_m if known_buffer_m is None else known_buffer_m),
        monument_buffer_m=profile.monument_buffer_m,
        closing_pixels=profile.closing_pixels,
        relief_scales_m=profile.relief_scales_m,
        pattern_scales_m=profile.pattern_scales_m,
        ml_sample_size=profile.ml_sample_size,
    )
    monument_extents = monument_extents or []
    modern_features = modern_features or []

    with rasterio.open(dtm_path) as dataset:
        data = dataset.read(1, masked=True).filled(np.nan).astype("float32")
        transform = dataset.transform
        x_resolution, y_resolution = float(dataset.res[0]), float(dataset.res[1])
        shape_out = data.shape
    valid = np.isfinite(data)
    filled = _fill_invalid(data)

    reliefs = calculate_local_relief_models(filled, x_resolution, y_resolution, profile.relief_scales_m, workers=workers)
    z_scores: dict[float, np.ndarray] = {}
    normalised_reliefs: dict[float, np.ndarray] = {}
    for scale_m, relief in reliefs.items():
        centre = float(np.median(relief))
        mad = float(np.median(np.abs(relief - centre)))
        scale = max(1e-6, 1.4826 * mad)
        z = np.abs((relief - centre) / scale).astype("float32")
        z[~valid] = 0.0
        z_scores[scale_m] = z
        normalised_reliefs[scale_m] = _robust_unit(z, 0.997)
    max_relief = np.max(np.stack(list(normalised_reliefs.values()), axis=0), axis=0)
    persistence_cutoff = max(1.25, 2.5 - profile.level * 0.08)
    persistence = np.mean(np.stack([(z >= persistence_cutoff).astype("float32") for z in z_scores.values()], axis=0), axis=0).astype("float32")

    linear_scales = tuple(scale for scale in (3.0, 5.0, 8.0, 12.0, 20.0, 32.0) if scale <= max(profile.pattern_scales_m))
    linear_response, linear_scale = calculate_linear_anomaly_response(filled, x_resolution, y_resolution, linear_scales, workers=workers)
    linear_score = np.clip(linear_response, 0.0, 1.0).astype("float32")
    annular, ridge_valley, texture, coherence = _surface_pattern_maps(
        filled, x_resolution, y_resolution, reliefs, linear_score, profile.pattern_scales_m,
    )

    feature_channels = [max_relief, persistence, linear_score, annular, ridge_valley, texture, coherence]
    feature_channels.extend(normalised_reliefs.values())
    feature_matrix = np.stack([np.nan_to_num(channel, nan=0.0, posinf=1.0, neginf=0.0) for channel in feature_channels], axis=-1).astype("float32").reshape(-1, len(feature_channels))
    terrain_novelty, novelty_metadata = _terrain_novelty_map(
        feature_matrix,
        valid.reshape(-1),
        raster_shape=shape_out,
        sample_size=profile.ml_sample_size,
        workers=workers,
    )
    terrain_novelty = terrain_novelty.reshape(shape_out)

    discovery = (
        0.25 * max_relief + 0.13 * persistence + 0.16 * linear_score + 0.13 * annular
        + 0.11 * ridge_valley + 0.08 * texture + 0.10 * terrain_novelty + 0.04 * coherence
    ).astype("float32")
    discovery[~valid] = 0.0
    diagnostic_dir = diagnostic_raster_dir or dtm_path.parent
    diagnostic_outputs: dict[str, str] = {}
    for filename, values in (
        ("discovery-score.tif", discovery),
        ("terrain-novelty.tif", terrain_novelty),
        ("ring-response.tif", annular),
        ("ridge-valley-response.tif", ridge_valley),
    ):
        output = _write_diagnostic_raster(diagnostic_dir / filename, values, dtm_path)
        diagnostic_outputs[filename.removesuffix(".tif").replace("-", "_")] = str(output)
    if not valid.any():
        return [], {"name": "hybrid-terrain-pattern-detector", "version": "0.4.2", "status": "no-data"}

    threshold = float(np.percentile(discovery[valid], profile.threshold_percentile))
    seed_threshold = float(np.percentile(discovery[valid], profile.seed_percentile))
    linear_threshold = float(np.percentile(linear_score[valid], profile.linear_percentile))
    annular_threshold = float(np.percentile(annular[valid], profile.annular_percentile))
    novelty_threshold = float(np.percentile(terrain_novelty[valid], profile.novelty_percentile))
    straight_line_mask = _straight_line_support_mask(
        linear_score, x_resolution, y_resolution,
        threshold_percentile=profile.linear_percentile,
        min_line_length_m=profile.min_line_length_m,
        dilation_m=max(1.5, 3.0 - profile.level * 0.12),
    ) & valid
    pattern_seed = (
        (discovery >= seed_threshold)
        | (linear_score >= linear_threshold)
        | (annular >= annular_threshold)
        | (terrain_novelty >= novelty_threshold)
        | (ridge_valley >= max(0.55, 0.85 - profile.level * 0.025))
    ) & valid
    combined_mask = (
        (discovery >= threshold) | pattern_seed | straight_line_mask
    )

    detail_union = _geometry_union(aim_features, profile.detailed_buffer_m)
    monument_union = _geometry_union(monument_extents, profile.monument_buffer_m)
    detail_mask = _rasterise_geometry_mask(detail_union, shape_out, transform)
    monument_mask = _rasterise_geometry_mask(monument_union, shape_out, transform)
    exclusion_mask = detail_mask | monument_mask
    combined_mask &= ~exclusion_mask
    structure_size = max(3, profile.closing_pixels * 2 + 1)
    combined_mask = binary_closing(combined_mask, structure=np.ones((structure_size, structure_size), dtype=bool))
    combined_mask &= ~exclusion_mask
    combined_mask &= valid
    labels, _count = label(combined_mask, structure=np.ones((3, 3), dtype=np.uint8))
    # Avoid repeatedly scanning the entire AOI for every connected component.
    # At high sensitivity there can be hundreds of components, so global
    # ``labels == label_id`` masks turn candidate ranking into O(N * components).
    # Component sizes and bounding slices make the expensive feature reads local.
    component_sizes = np.bincount(labels.ravel())
    component_slices = find_objects(labels)

    modern_score = np.zeros(shape_out, dtype="float32")
    modern_metadata: dict[str, Any] = {"enabled": False}
    if modern_context_path is not None and modern_context_path.is_file():
        with rasterio.open(modern_context_path) as modern_raster:
            modern_score = np.nan_to_num(modern_raster.read(1).astype("float32"), nan=0.0)
        modern_metadata = {"enabled": True, "source": str(modern_context_path)}
    elif modern_features:
        temporary = dtm_path.parent / ".modern-context-score.tif"
        modern = build_modern_context_raster(dtm_path, modern_features, temporary)
        with rasterio.open(modern.score_path) as modern_raster:
            modern_score = np.nan_to_num(modern_raster.read(1).astype("float32"), nan=0.0)
        modern_metadata = {"enabled": True, **modern.metadata}
        temporary.unlink(missing_ok=True)
        temporary.with_suffix(temporary.suffix + ".json").unlink(missing_ok=True)

    satellite_score = np.zeros(shape_out, dtype="float32")
    satellite_metadata: dict[str, Any] = {"enabled": False}
    if satellite_support_path is not None and satellite_support_path.is_file():
        with rasterio.open(satellite_support_path) as sat:
            satellite_score = np.nan_to_num(sat.read(1).astype("float32"), nan=0.0)
        satellite_metadata = {"enabled": True, "source": str(satellite_support_path)}

    pixel_area = abs(x_resolution * y_resolution)
    candidate_geometries: dict[int, list[Any]] = {}
    for geometry_data, value in shapes(labels.astype("int32"), mask=labels > 0, transform=transform):
        candidate_geometries.setdefault(int(value), []).append(shape(geometry_data))

    he_signatures = _he_signatures(aim_features)
    provisional: list[TerrainCandidate] = []
    stats = {
        "components_considered": 0,
        "modern_heavily_penalised": 0,
        "known_outline_traces_rejected": 0,
        "kept_for_ranking": 0,
        "ring_seeded": 0,
        "linear_seeded": 0,
        "novelty_seeded": 0,
    }

    strongest_scale = np.zeros(shape_out, dtype="float32")
    strongest_relief = np.zeros(shape_out, dtype="float32")
    best_z = np.full(shape_out, -np.inf, dtype="float32")
    for scale_m, z_score in z_scores.items():
        selector = z_score > best_z
        strongest_scale[selector] = scale_m
        strongest_relief[selector] = reliefs[scale_m][selector]
        best_z[selector] = z_score[selector]

    for label_id, geometries in candidate_geometries.items():
        stats["components_considered"] += 1
        if label_id >= component_sizes.size:
            continue
        component_area = float(component_sizes[label_id]) * pixel_area
        if component_area < profile.min_area_m2 or component_area > profile.max_area_m2:
            continue

        component_slice = component_slices[label_id - 1] if 0 < label_id <= len(component_slices) else None
        if component_slice is None:
            continue
        ys, xs = component_slice
        local_labels = labels[component_slice]
        local_pixels = local_labels == label_id
        local_linear_mask = straight_line_mask[component_slice] & local_pixels

        # A Hough-supported line can sit inside a much larger terrain response
        # component (especially on sloping ground). For genuinely linear signals,
        # rank the line corridor itself rather than allowing the surrounding slope
        # to dominate its geometry. All expensive raster operations stay inside
        # the component's bounding window.
        line_pixel_count = int(local_linear_mask.sum())
        local_pixel_count = int(local_pixels.sum())
        line_area = float(line_pixel_count) * pixel_area
        use_line = (
            line_area >= profile.min_area_m2
            and line_pixel_count >= max(20, int(local_pixel_count * 0.08))
        )
        score_pixels = local_linear_mask if use_line else local_pixels

        if use_line:
            from rasterio.windows import Window, transform as window_transform

            local_window = Window.from_slices(ys, xs)
            local_transform = window_transform(local_window, transform)
            local_geometries = [
                shape(geometry_data)
                for geometry_data, value in shapes(
                    score_pixels.astype("uint8"),
                    mask=score_pixels,
                    transform=local_transform,
                )
                if int(value) == 1
            ]
            geometries = local_geometries or geometries

        geometry = unary_union(geometries).buffer(0)
        if detail_union is not None:
            geometry = geometry.difference(detail_union).buffer(0)
        if monument_union is not None:
            geometry = geometry.difference(monument_union).buffer(0)
        if geometry.is_empty or float(geometry.area) < profile.min_area_m2:
            continue
        if _near_parallel_known_feature(geometry, aim_features):
            stats["known_outline_traces_rejected"] += 1
            continue

        discovery_local = discovery[component_slice]
        strongest_relief_local = strongest_relief[component_slice]
        strongest_scale_local = strongest_scale[component_slice]
        max_relief_local = max_relief[component_slice]
        modern_local = modern_score[component_slice]
        satellite_local = satellite_score[component_slice]
        linear_local = linear_score[component_slice]
        annular_local = annular[component_slice]
        novelty_local = terrain_novelty[component_slice]
        ridge_local = ridge_valley[component_slice]
        texture_local = texture[component_slice]

        discovery_strength = float(np.quantile(discovery_local[score_pixels], 0.82))
        relief_values = np.abs(strongest_relief_local[score_pixels])
        strongest_index = int(np.argmax(relief_values)) if relief_values.size else 0
        raw_relief = float(strongest_relief_local[score_pixels][strongest_index]) if relief_values.size else 0.0
        morphology = _geometry_metrics(geometry)
        linear_tail = float(np.clip(_raster_tail_mean_values(linear_local, score_pixels, 0.72), 0.0, 1.0))
        annular_tail = float(np.clip(_raster_tail_mean_values(annular_local, score_pixels, 0.72), 0.0, 1.0))
        novelty_tail = float(np.clip(_raster_tail_mean_values(novelty_local, score_pixels, 0.72), 0.0, 1.0))
        ridge_tail = float(np.clip(_raster_tail_mean_values(ridge_local, score_pixels, 0.72), 0.0, 1.0))
        texture_tail = float(np.clip(_raster_tail_mean_values(texture_local, score_pixels, 0.72), 0.0, 1.0))
        lidar_strength = float(np.clip(np.quantile(max_relief_local[score_pixels], 0.85), 0.0, 1.0))
        persistence_score = float(np.mean(np.stack([(z_scores[scale][component_slice][score_pixels] >= persistence_cutoff).astype("float32") for scale in sorted(z_scores)], axis=0)))
        shape_score = _morphology_score(morphology, linear_tail)
        modern_penalty = float(np.clip(_raster_tail_mean_values(modern_local, score_pixels), 0.0, 1.0))
        satellite_support = float(np.clip(np.mean(satellite_local[score_pixels]) if np.any(score_pixels) else 0.0, 0.0, 1.0))
        he_similarity = _he_similarity(morphology, he_signatures)

        linear_bonus = 0.05 if linear_tail >= 0.65 and morphology["elongation"] >= 7.0 else 0.0
        ring_bonus = 0.07 if annular_tail >= 0.65 and morphology["circularity"] >= 0.42 else 0.0
        novelty_bonus = 0.05 if novelty_tail >= 0.70 else 0.0
        context_weight = 1.0 - 0.74 * modern_penalty
        base = (
            0.28 * discovery_strength + 0.15 * lidar_strength + 0.12 * persistence_score
            + 0.10 * shape_score + 0.10 * linear_tail + 0.10 * annular_tail
            + 0.07 * ridge_tail + 0.08 * novelty_tail + 0.04 * texture_tail
            + 0.03 * satellite_support + 0.02 * he_similarity
            + linear_bonus + ring_bonus + novelty_bonus
        )
        final_score = 100.0 * max(0.0, base * context_weight)

        reasons: list[str] = []
        if shape_score >= 0.63: reasons.append("coherent archaeological morphology")
        if morphology["holes"] > 0 or (annular_tail >= 0.62 and morphology["circularity"] >= 0.35): reasons.append("ring/bank morphology")
        if persistence_score >= 0.34: reasons.append("persists across multiple LiDAR scales")
        if linear_tail >= 0.52: reasons.append("strong coherent linear response")
        if morphology["elongation"] >= 8.0: reasons.append("elongated linear morphology")
        if ridge_tail >= 0.60: reasons.append("ridge/valley relief signature")
        if novelty_tail >= 0.58: reasons.append("unusual terrain pattern learned within this AOI")
        if texture_tail >= 0.58: reasons.append("distinctive local terrain texture")
        if satellite_support >= 0.20: reasons.append("satellite vegetation/reflectance anomaly supports the terrain signal")
        if he_similarity >= 0.75: reasons.append("geometry resembles known Historic England mapped forms")
        if modern_penalty >= 0.65:
            reasons.append("strong modern-feature conflict")
            stats["modern_heavily_penalised"] += 1
        if annular_tail >= annular_threshold: stats["ring_seeded"] += 1
        if linear_tail >= linear_threshold: stats["linear_seeded"] += 1
        if novelty_tail >= novelty_threshold: stats["novelty_seeded"] += 1
        stats["kept_for_ranking"] += 1

        provisional.append(TerrainCandidate(
            candidate_id=len(provisional) + 1,
            geometry=geometry,
            score=final_score,
            area_m2=float(geometry.area),
            relief_m=float(np.max(relief_values)) if relief_values.size else 0.0,
            polarity="positive" if raw_relief >= 0 else "negative",
            strongest_scale_m=float(strongest_scale_local[score_pixels][strongest_index]) if relief_values.size else 0.0,
            lidar_score=100.0 * lidar_strength,
            persistence_score=100.0 * persistence_score,
            morphology_score=100.0 * shape_score,
            modern_penalty=100.0 * modern_penalty,
            satellite_support=100.0 * satellite_support,
            he_similarity=100.0 * he_similarity,
            classification="candidate",
            reasons=tuple(reasons),
            morphology=tuple(sorted((key, float(value)) for key, value in morphology.items())),
            linear_score=100.0 * linear_tail,
            terrain_novelty_score=100.0 * novelty_tail,
            ring_score=100.0 * annular_tail,
            ridge_valley_score=100.0 * ridge_tail,
            texture_score=100.0 * texture_tail,
        ))

    provisional.sort(key=lambda item: (item.score, item.terrain_novelty_score, item.linear_score, item.ring_score, item.relief_m), reverse=True)
    retained: list[TerrainCandidate] = []
    rejected_modern = 0
    for candidate in provisional:
        morphology = dict(candidate.morphology)
        if candidate.modern_penalty >= 97.0 and morphology.get("elongation", 0.0) < 6.0 and candidate.morphology_score < 55.0:
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
            classification="high-priority" if item.score >= (74.0 - profile.level * 0.8) else "candidate",
            reasons=item.reasons,
            morphology=item.morphology,
            linear_score=item.linear_score,
            terrain_novelty_score=item.terrain_novelty_score,
            ring_score=item.ring_score,
            ridge_valley_score=item.ridge_valley_score,
            texture_score=item.texture_score,
        )
        for index, item in enumerate(retained, 1)
    ]

    metadata = {
        "name": "hybrid-terrain-pattern-detector",
        "version": "0.4.2",
        "sensitivity": profile.level,
        "sensitivity_description": "1=very conservative, 5=balanced research setting, 10=maximum exploratory recall",
        "workers_requested": workers,
        "workers_auto_cap": 6,
        "scales_m": list(reliefs),
        "discovery_channels": {
            "local_relief": 0.25, "persistence": 0.13, "linear": 0.16,
            "annular_ring": 0.13, "ridge_valley": 0.11, "texture": 0.08,
            "terrain_novelty_ml": 0.10, "surface_coherence": 0.04,
        },
        "machine_learning": novelty_metadata,
        "threshold_percentile": profile.threshold_percentile,
        "seed_percentile": profile.seed_percentile,
        "linear_response_percentile": profile.linear_percentile,
        "annular_seed_percentile": profile.annular_percentile,
        "novelty_seed_percentile": profile.novelty_percentile,
        "threshold": threshold,
        "linear_threshold": linear_threshold,
        "annular_threshold": annular_threshold,
        "novelty_threshold": novelty_threshold,
        "straight_line_seeded": True,
        "minimum_hough_line_length_m": profile.min_line_length_m,
        "min_area_m2": profile.min_area_m2,
        "max_area_m2": profile.max_area_m2,
        "max_candidates": profile.max_candidates,
        "provisional_candidate_count": len(provisional),
        "candidate_count": len(retained),
        "rejected_for_modern_context": rejected_modern,
        "modern_context": modern_metadata,
        "satellite_context": satellite_metadata,
        "diagnostic_rasters": diagnostic_outputs,
        "historic_england_reference": {
            "detail": "Historic England is contextual evidence and an exclusion mask; candidate generation remains valid with zero HE records.",
            "detailed_mapping_count": len(aim_features),
            "monument_extents_count": len(monument_extents),
            "monument_extents_exclusion": bool(monument_extents),
            "monument_extents_buffer_m": profile.monument_buffer_m,
            "detailed_mapping_exclusion_buffer_m": profile.detailed_buffer_m,
            "candidate_generation_requires_historic_england": False,
        },
        "provisional_statistics": stats,
        "warning": (
            "The detector combines deterministic terrain morphology with unsupervised terrain novelty. "
            "The ML component learns unusual patterns within this AOI but has no archaeological training labels yet. "
            "Candidates remain research leads, not confirmed archaeological identifications."
        ),
    }
    return retained, metadata
