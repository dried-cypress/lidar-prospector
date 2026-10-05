from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_feature_collection(features: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_candidate_collection(candidates: Iterable[Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for candidate in candidates:
        features.append(
            {
                "type": "Feature",
                "geometry": candidate.geometry.__geo_interface__,
                "properties": {
                    "candidate_id": candidate.candidate_id,
                    "classification": candidate.classification,
                    "score": candidate.score,
                    "area_m2": candidate.area_m2,
                    "relief_m": candidate.relief_m,
                    "polarity": candidate.polarity,
                    "strongest_scale_m": candidate.strongest_scale_m,
                    "lidar_score": candidate.lidar_score,
                    "persistence_score": candidate.persistence_score,
                    "morphology_score": candidate.morphology_score,
                    "modern_penalty": candidate.modern_penalty,
                    "satellite_support": candidate.satellite_support,
                    "he_similarity": candidate.he_similarity,
                    "trained_model_score": getattr(candidate, "trained_model_score", 0.0),
                    "he_match_type": getattr(candidate, "he_match_type", ""),
                    "he_match_uid": getattr(candidate, "he_match_uid", ""),
                    "reasons": list(candidate.reasons),
                    "morphology": dict(candidate.morphology),
                },
            }
        )
    write_feature_collection(features, destination)
