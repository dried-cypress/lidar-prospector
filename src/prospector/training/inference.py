from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from prospector.training.features import DEFAULT_SCALES_M, patch_from_raster
from prospector.training.model import load_model, predict_descriptor


def score_candidates(
    model_path: Path,
    raster_path: Path,
    candidates: list[Any],
    *,
    scales: tuple[float, ...] = DEFAULT_SCALES_M,
) -> tuple[list[Any], dict[str, Any]]:
    """Score candidate objects against the persisted archaeology model.

    Multiple physical scales are tested at every candidate centre. The best
    scale is retained, making the classifier substantially less sensitive to
    monument size than a single fixed-size patch.
    """
    model, model_meta = load_model(model_path)
    scored: list[Any] = []
    for candidate in candidates:
        point = candidate.geometry.representative_point()
        area = max(float(candidate.area_m2), 1.0)
        equivalent = 2.0 * np.sqrt(area / np.pi)
        base_scale = float(np.clip(max(equivalent * 2.5, 32.0), 32.0, 256.0))
        test_scales = tuple(float(np.clip(base_scale * factor, 32.0, 256.0)) for factor in (0.75, 1.0, 1.5))
        test_scales = tuple(dict.fromkeys(test_scales)) or scales
        probabilities: list[tuple[float, float]] = []
        for scale_m in test_scales:
            descriptor = patch_from_raster(
                raster_path,
                float(point.x),
                float(point.y),
                scale_m,
            )
            probabilities.append((predict_descriptor(model, descriptor), scale_m))
        probability, scale_m = max(probabilities, key=lambda item: item[0])
        combined = float(np.clip(0.58 * candidate.score / 100.0 + 0.42 * probability, 0.0, 1.0)) * 100.0
        reasons = list(candidate.reasons)
        if probability >= 0.80:
            reasons.append(f"trained model strongly matches archaeological terrain ({probability * 100.0:.0f}%)")
        elif probability >= 0.62:
            reasons.append(f"trained model supports archaeological terrain ({probability * 100.0:.0f}%)")
        scored.append(replace(
            candidate,
            score=combined,
            reasons=tuple(reasons),
            trained_model_score=probability * 100.0,
        ))
    scored.sort(key=lambda item: (item.score, item.trained_model_score), reverse=True)
    return [
        replace(item, candidate_id=index)
        for index, item in enumerate(scored, 1)
    ], {
        "enabled": True,
        "model": model_meta.get("model_name", "prospector-archaeology"),
        "version": model_meta.get("version"),
        "feature_version": model_meta.get("feature_version"),
        "multi_scale_inference": True,
        "scales_per_candidate": [0.75, 1.0, 1.5],
        "candidates_scored": len(scored),
    }
