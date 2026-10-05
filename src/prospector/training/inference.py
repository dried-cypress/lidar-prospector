from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from prospector.training.features import (
    DEFAULT_SCALES_M,
    FEATURE_VERSION,
    descriptor_from_dataset,
    patch_from_raster,
)
from prospector.training.model import load_model, predict_descriptor


def _runtime_model_check(metadata: dict[str, Any]) -> None:
    import sklearn

    expected = str(metadata.get("sklearn_version") or "")
    if not expected:
        raise RuntimeError(
            "Persisted archaeology model has no scikit-learn version metadata. "
            "Retrain the model with the current Prospector release before inference."
        )
    if expected != sklearn.__version__:
        raise RuntimeError(
            f"Persisted archaeology model was trained with scikit-learn {expected}, "
            f"but this runtime uses {sklearn.__version__}. Retrain the model with the "
            "current Prospector release."
        )
    feature_version = str(metadata.get("feature_version") or "")
    if feature_version != FEATURE_VERSION:
        raise RuntimeError(
            f"Persisted archaeology model uses feature schema {feature_version or 'unknown'}, "
            f"but this runtime requires {FEATURE_VERSION}. Rebuild the training dataset "
            "and retrain the model with the current Prospector release."
        )


def _prototype_probability(model: Any, descriptors: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    positive = np.asarray(metadata.get("positive_prototype") or [], dtype="float32")
    negative = np.asarray(metadata.get("negative_prototype") or [], dtype="float32")
    if positive.size == 0 or negative.size == 0:
        return np.full((descriptors.shape[0],), 0.5, dtype="float32")
    scaler = model.named_steps["scale"]
    pca = model.named_steps["pca"]
    embedded = pca.transform(scaler.transform(descriptors)).astype("float32")
    embedded /= np.maximum(np.linalg.norm(embedded, axis=1, keepdims=True), 1e-8)
    positive_similarity = embedded @ positive
    negative_similarity = embedded @ negative
    margin = np.clip(positive_similarity - negative_similarity, -2.0, 2.0)
    return (1.0 / (1.0 + np.exp(-4.0 * margin))).astype("float32")


def predict_descriptors(model: Any, descriptors: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    classifier_probability = model.predict_proba(descriptors)[:, 1].astype("float32")
    prototype_probability = _prototype_probability(model, descriptors, metadata)
    return np.clip(0.68 * classifier_probability + 0.32 * prototype_probability, 0.0, 1.0).astype("float32")


def predict_descriptor(model: Any, descriptor: np.ndarray, metadata: dict[str, Any] | None = None) -> float:
    return float(predict_descriptors(model, descriptor.reshape(1, -1), metadata or {})[0])


def score_candidates(
    model_path: Path,
    raster_path: Path,
    candidates: list[Any],
    *,
    scales: tuple[float, ...] = DEFAULT_SCALES_M,
) -> tuple[list[Any], dict[str, Any]]:
    """Score candidate objects against the persisted archaeology model."""
    model, model_meta = load_model(model_path)
    _runtime_model_check(model_meta)
    scored: list[Any] = []
    for candidate in candidates:
        point = candidate.geometry.representative_point()
        area = max(float(candidate.area_m2), 1.0)
        equivalent = 2.0 * np.sqrt(area / np.pi)
        base_scale = float(np.clip(max(equivalent * 2.5, 32.0), 32.0, 256.0))
        test_scales = tuple(
            float(np.clip(base_scale * factor, 32.0, 256.0)) for factor in (0.75, 1.0, 1.5)
        )
        test_scales = tuple(dict.fromkeys(test_scales)) or scales
        probabilities: list[tuple[float, float]] = []
        for scale_m in test_scales:
            descriptor = patch_from_raster(raster_path, float(point.x), float(point.y), scale_m)
            probabilities.append((predict_descriptor(model, descriptor, model_meta), scale_m))
        probability, scale_m = max(probabilities, key=lambda item: item[0])
        combined = float(
            np.clip(0.42 * candidate.score / 100.0 + 0.58 * probability, 0.0, 1.0)
        ) * 100.0
        reasons = list(candidate.reasons)
        if probability >= 0.80:
            reasons.append(f"trained model strongly matches archaeological terrain ({probability * 100.0:.0f}%)")
        elif probability >= 0.62:
            reasons.append(f"trained model supports archaeological terrain ({probability * 100.0:.0f}%)")
        scored.append(
            replace(
                candidate,
                score=combined,
                reasons=tuple(reasons),
                trained_model_score=probability * 100.0,
            )
        )
    scored.sort(key=lambda item: (item.score, item.trained_model_score), reverse=True)
    return [replace(item, candidate_id=index) for index, item in enumerate(scored, 1)], {
        "enabled": True,
        "model": model_meta.get("model_name", "prospector-archaeology"),
        "version": model_meta.get("version"),
        "feature_version": model_meta.get("feature_version"),
        "sklearn_version": model_meta.get("sklearn_version"),
        "prototype_version": model_meta.get("prototype_version"),
        "classifier_weight": 0.68,
        "prototype_weight": 0.32,
        "multi_scale_inference": True,
        "scales_per_candidate": [0.75, 1.0, 1.5],
        "candidate_generation": True,
        "candidates_scored": len(scored),
    }


def predict_raster_likelihood(
    model_path: Path,
    raster_path: Path,
    *,
    level: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project the persisted archaeology model across the LiDAR AOI.

    Inference is performed on a coarse physical grid and interpolated, rather
    than evaluating every DTM cell. The model therefore participates in
    candidate generation instead of merely rescoring heuristic detections.
    """
    import rasterio
    from scipy.ndimage import zoom

    model, model_meta = load_model(model_path)
    _runtime_model_check(model_meta)
    level = max(1, min(10, int(level)))
    with rasterio.open(raster_path) as dataset:
        height, width = dataset.height, dataset.width
        resolution = max(abs(float(dataset.res[0])), abs(float(dataset.res[1])))
        step_m = max(8.0, 24.0 - level * 1.6)
        step_px = max(1, int(round(step_m / resolution)))
        xs = np.arange(0, width, step_px, dtype=int)
        ys = np.arange(0, height, step_px, dtype=int)
        scales = (32.0, 64.0, 128.0, 192.0, 256.0) if level >= 8 else (48.0, 96.0, 192.0)
        coarse = np.zeros((len(ys), len(xs)), dtype="float32")
        descriptors: list[np.ndarray] = []
        locations: list[tuple[int, int]] = []
        x0, y0 = dataset.transform * (0, 0)
        for iy, row in enumerate(ys):
            for ix, col in enumerate(xs):
                x, y = dataset.transform * (int(col) + 0.5, int(row) + 0.5)
                best = 0.0
                for scale_m in scales:
                    descriptors.append(
                        descriptor_from_dataset(dataset, float(x), float(y), scale_m)
                    )
                    locations.append((iy, ix))
                # best-of-scales is resolved in batches below; retain all descriptors.
        batch_size = 128
        probabilities: list[float] = []
        for start in range(0, len(descriptors), batch_size):
            batch = np.stack(descriptors[start:start + batch_size]).astype("float32")
            probabilities.extend(predict_descriptors(model, batch, model_meta).tolist())
        for group_start in range(0, len(locations), len(scales)):
            group = probabilities[group_start:group_start + len(scales)]
            iy, ix = locations[group_start]
            coarse[iy, ix] = max(group) if group else 0.0

    if coarse.size == 0:
        return np.zeros((height, width), dtype="float32"), {"enabled": False, "reason": "empty_grid"}
    likelihood = zoom(
        coarse,
        (height / coarse.shape[0], width / coarse.shape[1]),
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype("float32")[:height, :width]
    return likelihood, {
        "enabled": True,
        "model": model_meta.get("model_name", "prospector-archaeology"),
        "version": model_meta.get("version"),
        "feature_version": model_meta.get("feature_version"),
        "sklearn_version": model_meta.get("sklearn_version"),
        "prototype_version": model_meta.get("prototype_version"),
        "classifier_weight": 0.68,
        "prototype_weight": 0.32,
        "spatial_step_m": step_m,
        "spatial_step_pixels": step_px,
        "grid_width": int(coarse.shape[1]),
        "grid_height": int(coarse.shape[0]),
        "scales_m": list(scales),
        "predictions": len(descriptors),
        "candidate_generation_input": True,
    }
