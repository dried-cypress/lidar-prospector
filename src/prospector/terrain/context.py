from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ModernContextRasters:
    score_path: Path
    masks: dict[str, np.ndarray]
    metadata: dict[str, Any]


MODERN_WEIGHTS: dict[str, tuple[float, float]] = {
    # (maximum penalty, e-folding distance in metres)
    "building": (1.00, 8.0),
    "road": (0.98, 7.0),
    "track": (0.90, 5.0),
    "path": (0.55, 3.5),
    "boundary": (0.65, 3.0),
    "water": (0.88, 5.0),
}


def build_modern_context_raster(
    dtm_path: Path,
    features: list[dict[str, Any]],
    output_path: Path,
) -> ModernContextRasters:
    """Rasterise current mapped features into a soft modernity penalty.

    Roads and buildings receive the strongest suppression; paths and boundaries
    are deliberately softer because historic features can survive beneath or
    alongside modern access and land boundaries.
    """
    import rasterio
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt
    from shapely.geometry import shape

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dtm_path) as dataset:
        shape_out = (dataset.height, dataset.width)
        transform = dataset.transform
        resolution = (abs(float(dataset.res[1])), abs(float(dataset.res[0])))
        profile = dataset.profile.copy()

    masks: dict[str, np.ndarray] = {
        name: np.zeros(shape_out, dtype=bool) for name in MODERN_WEIGHTS
    }
    for feature in features:
        properties = feature.get("properties") or {}
        context_type = str(properties.get("prospector_context_type") or "").casefold()
        if context_type not in masks:
            continue
        geometry_data = feature.get("geometry")
        if not geometry_data:
            continue
        try:
            geometry = shape(geometry_data)
        except Exception:
            continue
        if geometry.is_empty:
            continue
        layer_mask = rasterize(
            [(geometry, 1)],
            out_shape=shape_out,
            transform=transform,
            fill=0,
            default_value=1,
            dtype="uint8",
            all_touched=True,
        ).astype(bool)
        masks[context_type] |= layer_mask

    score = np.zeros(shape_out, dtype="float32")
    for context_type, (weight, _decay) in MODERN_WEIGHTS.items():
        mask = masks[context_type]
        if not mask.any():
            continue
        distance = distance_transform_edt(~mask, sampling=resolution)
        contribution = weight * np.exp(-distance / _decay)
        score = np.maximum(score, contribution.astype("float32"))

    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(score, 1)

    metadata = {
        "feature_count": len(features),
        "context_types": {
            name: int(mask.sum()) for name, mask in masks.items()
        },
        "weights": {
            name: {"max_penalty": values[0], "decay_m": values[1]}
            for name, values in MODERN_WEIGHTS.items()
        },
    }
    return ModernContextRasters(output_path, masks, metadata)
