from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "prospector-archaeology"
MODEL_VERSION_PREFIX = "0.5"


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_path: Path
    metadata_path: Path
    version: str
    metrics: dict[str, float]


def train_model(
    dataset_path: Path,
    model_dir: Path,
    *,
    random_state: int = 20261005,
) -> TrainingResult:
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    payload = np.load(dataset_path, allow_pickle=False)
    x = payload["X"].astype("float32")
    y = payload["y"].astype("uint8")
    groups = payload["groups"].astype(str)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[1] == 0:
        raise ValueError("Training dataset has inconsistent feature/label shapes")
    if np.unique(y).size < 2:
        raise ValueError("Training requires both positive archaeology and background examples")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    try:
        train_idx, test_idx = next(splitter.split(x, y, groups))
    except ValueError:
        # Small single-group datasets cannot be spatially held out. Fit the model,
        # but record that no honest hold-out metric is available.
        train_idx = np.arange(x.shape[0])
        test_idx = np.array([], dtype=int)

    n_components = min(96, x.shape[1], max(8, x.shape[0] // 10))
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, whiten=True, random_state=random_state, svd_solver="randomized")),
        ("classifier", ExtraTreesClassifier(
            n_estimators=300,
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )),
    ])
    pipeline.fit(x[train_idx], y[train_idx])

    metrics: dict[str, float] = {
        "training_examples": float(len(train_idx)),
        "testing_examples": float(len(test_idx)),
        "positive_rate": float(np.mean(y)),
        "holdout_available": float(bool(test_idx.size)),
    }
    if test_idx.size:
        pred = pipeline.predict(x[test_idx])
        probability = pipeline.predict_proba(x[test_idx])[:, 1]
        metrics.update({
            "accuracy": float(accuracy_score(y[test_idx], pred)),
            "precision": float(precision_score(y[test_idx], pred, zero_division=0)),
            "recall": float(recall_score(y[test_idx], pred, zero_division=0)),
            "f1": float(f1_score(y[test_idx], pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y[test_idx], probability)) if np.unique(y[test_idx]).size == 2 else 0.0,
        })

    model_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"{MODEL_VERSION_PREFIX}.{stamp}"
    model_path = model_dir / f"{MODEL_NAME}-{stamp}.joblib"
    metadata_path = model_dir / f"{MODEL_NAME}-{stamp}.json"
    payload_meta: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_version": "patch-v1",
        "model": "StandardScaler + PCA + ExtraTreesClassifier",
        "training_dataset": str(dataset_path),
        "random_state": random_state,
        "rotation_augmented": True,
        "multi_scale": True,
        "metrics": metrics,
        "classes": {"0": "background", "1": "archaeology"},
    }
    joblib.dump(pipeline, model_path, compress=3)
    metadata_path.write_text(json.dumps(payload_meta, indent=2, sort_keys=True), encoding="utf-8")
    current = model_dir / "current.joblib"
    current_meta = model_dir / "current.json"
    joblib.dump(pipeline, current, compress=3)
    current_meta.write_text(json.dumps(payload_meta, indent=2, sort_keys=True), encoding="utf-8")
    return TrainingResult(model_path, metadata_path, version, metrics)


def load_model(model_path: Path) -> tuple[Any, dict[str, Any]]:
    import joblib
    path = model_path
    metadata_path = path.with_suffix(".json")
    if path.name == "current.joblib":
        metadata_path = path.with_name("current.json")
    model = joblib.load(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    return model, metadata


def predict_descriptor(model: Any, descriptor: np.ndarray) -> float:
    probability = model.predict_proba(descriptor.reshape(1, -1))[:, 1]
    return float(probability[0])
