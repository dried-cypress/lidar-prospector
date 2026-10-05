from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_VERSION = "patch-v1"
PATCH_PIXELS = 32
DEFAULT_SCALES_M = (32.0, 64.0, 128.0)
DEFAULT_ROTATIONS = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)


@dataclass(frozen=True, slots=True)
class PatchDescriptor:
    vector: np.ndarray
    scale_m: float
    rotation_deg: float


def _robust_normalise(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype="float32")
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype="float32")
    median = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - median)))
    scale = max(0.05, 1.4826 * mad)
    return np.clip((values - median) / scale, -6.0, 6.0).astype("float32")


def _resize(array: np.ndarray, size: int = PATCH_PIXELS) -> np.ndarray:
    from skimage.transform import resize
    return resize(
        array,
        (size, size),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype("float32")


def _channels(elevation: np.ndarray, x_resolution: float, y_resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter
    gy, gx = np.gradient(elevation, y_resolution, x_resolution)
    slope = np.hypot(gx, gy).astype("float32")
    smooth = gaussian_filter(elevation, sigma=1.0, mode="nearest")
    laplacian = (
        np.gradient(np.gradient(smooth, axis=1), axis=1)
        + np.gradient(np.gradient(smooth, axis=0), axis=0)
    ).astype("float32")
    return elevation.astype("float32"), slope, laplacian


def descriptor_from_array(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
    *,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    from scipy.ndimage import rotate

    filled = np.asarray(elevation, dtype="float32")
    if not np.isfinite(filled).all():
        finite = filled[np.isfinite(filled)]
        fill = float(np.median(finite)) if finite.size else 0.0
        filled = np.nan_to_num(filled, nan=fill, posinf=fill, neginf=fill)
    elevation_channel, slope, curvature = _channels(filled, x_resolution, y_resolution)
    channels = [_resize(_robust_normalise(channel)) for channel in (elevation_channel, slope, curvature)]
    if rotation_deg:
        channels = [
            rotate(channel, rotation_deg, reshape=False, order=1, mode="reflect", prefilter=False).astype("float32")
            for channel in channels
        ]
    radial = _radial_signature(channels[0])
    oriented = _orientation_invariant_moments(channels[0])
    return np.concatenate([*(channel.ravel() for channel in channels), radial, oriented]).astype("float32")


def _radial_signature(image: np.ndarray, bins: int = 16) -> np.ndarray:
    size_y, size_x = image.shape
    yy, xx = np.mgrid[:size_y, :size_x]
    cy = (size_y - 1) / 2.0
    cx = (size_x - 1) / 2.0
    radius = np.hypot(xx - cx, yy - cy)
    normalised = radius / max(radius.max(), 1e-6)
    result = np.zeros(bins, dtype="float32")
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        mask = (normalised >= lo) & (normalised < hi)
        result[index] = float(np.mean(image[mask])) if mask.any() else 0.0
    return result


def _orientation_invariant_moments(image: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]].astype("float32")
    weights = np.abs(image).astype("float32") + 1e-3
    total = float(weights.sum())
    if total <= 0.0:
        return np.zeros(8, dtype="float32")
    cx = float((xx * weights).sum() / total)
    cy = float((yy * weights).sum() / total)
    dx = xx - cx
    dy = yy - cy
    mu20 = float((dx * dx * weights).sum() / total)
    mu02 = float((dy * dy * weights).sum() / total)
    mu11 = float((dx * dy * weights).sum() / total)
    covariance = np.array([[mu20, mu11], [mu11, mu02]], dtype="float64")
    eigenvalues = np.linalg.eigvalsh(covariance)
    trace = float(np.trace(covariance))
    det = float(np.linalg.det(covariance))
    skew_r = float(np.mean((dx**3 + dy**3) * weights) / max(total, 1e-6))
    kurt_r = float(np.mean((dx**4 + dy**4) * weights) / max(total, 1e-6))
    return np.asarray([
        float(eigenvalues[0]), float(eigenvalues[1]), trace, det,
        float(eigenvalues[1] / max(eigenvalues[0], 1e-6)), skew_r, kurt_r,
        float(np.mean(weights)),
    ], dtype="float32")


def patch_from_raster(
    raster_path: Path,
    centre_x: float,
    centre_y: float,
    scale_m: float,
    *,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(raster_path) as dataset:
        half = scale_m / 2.0
        window = from_bounds(
            centre_x - half,
            centre_y - half,
            centre_x + half,
            centre_y + half,
            transform=dataset.transform,
        )
        array = dataset.read(
            1,
            window=window,
            boundless=True,
            fill_value=np.nan,
            out_shape=(max(64, PATCH_PIXELS * 2), max(64, PATCH_PIXELS * 2)),
            resampling=rasterio.enums.Resampling.bilinear,
        ).astype("float32")
        return descriptor_from_array(
            array,
            dataset.res[0],
            dataset.res[1],
            rotation_deg=rotation_deg,
        )


def feature_vector_size() -> int:
    return 3 * PATCH_PIXELS * PATCH_PIXELS + 16 + 8
