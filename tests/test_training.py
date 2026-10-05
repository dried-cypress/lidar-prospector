from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prospector.training.features import DEFAULT_ROTATIONS, descriptor_from_array, feature_vector_size
from prospector.training.model import train_model
from prospector.training.pipeline import _feature_scale, _grid


def test_training_descriptor_is_fixed_size_and_finite() -> None:
    y, x = np.mgrid[-1:1:33j, -1:1:33j]
    patch = np.exp(-(x * x + y * y) * 6.0).astype("float32")
    descriptor = descriptor_from_array(patch, 1.0, 1.0)
    assert descriptor.shape == (feature_vector_size(),)
    assert np.isfinite(descriptor).all()


def test_rotation_descriptor_keeps_same_feature_dimensions() -> None:
    patch = np.zeros((64, 64), dtype="float32")
    patch[28:36, 10:54] = 1.0
    base = descriptor_from_array(patch, 1.0, 1.0, rotation_deg=0.0)
    rotated = descriptor_from_array(patch, 1.0, 1.0, rotation_deg=90.0)
    assert base.shape == rotated.shape == (feature_vector_size(),)
    assert np.isfinite(rotated).all()
    assert not np.allclose(base, rotated)
    assert DEFAULT_ROTATIONS == (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)


def test_grid_tiles_cover_requested_bbox_without_gaps() -> None:
    tiles = _grid((0.0, 0.0, 2500.0, 1800.0), 1000.0)
    assert len(tiles) == 6
    assert min(tile[0] for tile in tiles) == 0.0
    assert max(tile[2] for tile in tiles) == 2500.0
    assert min(tile[1] for tile in tiles) == 0.0
    assert max(tile[3] for tile in tiles) == 1800.0


def test_feature_scale_handles_small_and_large_shapes() -> None:
    from shapely.geometry import box
    assert _feature_scale({"geometry": box(0, 0, 4, 4).__geo_interface__}) == 32.0
    assert _feature_scale({"geometry": box(0, 0, 200, 200).__geo_interface__}) == 256.0


def test_train_model_requires_two_classes(tmp_path: Path) -> None:
    dataset = tmp_path / "training.npz"
    x = np.zeros((8, 10), dtype="float32")
    y = np.ones(8, dtype="uint8")
    groups = np.asarray([f"g{i}" for i in range(8)])
    np.savez_compressed(dataset, X=x, y=y, groups=groups)
    with pytest.raises(ValueError, match="both positive archaeology and background"):
        train_model(dataset, tmp_path / "models")


def test_train_model_creates_versioned_and_current_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    positive = rng.normal(1.0, 0.2, size=(24, 16)).astype("float32")
    negative = rng.normal(-1.0, 0.2, size=(24, 16)).astype("float32")
    x = np.vstack([positive, negative])
    y = np.asarray([1] * len(positive) + [0] * len(negative), dtype="uint8")
    groups = np.asarray([f"p{i//3}" for i in range(24)] + [f"n{i//3}" for i in range(24)])
    dataset = tmp_path / "training.npz"
    np.savez_compressed(dataset, X=x, y=y, groups=groups)

    result = train_model(dataset, tmp_path / "models")
    assert result.feature_dimensions == 16
    assert result.model_path.is_file()
    assert result.metadata_path.is_file()
    assert (tmp_path / "models" / "current.joblib").is_file()
    assert (tmp_path / "models" / "current.json").is_file()
    metadata = json.loads((tmp_path / "models" / "current.json").read_text())
    assert metadata["model_name"] == "prospector-archaeology"
    assert metadata["rotation_augmented"] is True
    assert metadata["multi_scale"] is True


def test_training_schema_contains_postgis_and_registry() -> None:
    from prospector.training.db import SCHEMA_SQL
    assert "CREATE EXTENSION IF NOT EXISTS postgis" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS aim_features" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS model_registry" in SCHEMA_SQL


def test_persisted_model_scores_a_real_terrain_candidate(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from prospector.terrain.anomalies import TerrainCandidate
    from prospector.training.inference import score_candidates

    raster = tmp_path / "candidate.tif"
    yy, xx = np.mgrid[0:96, 0:96]
    patch = (np.exp(-(((xx - 48) ** 2 + (yy - 48) ** 2) / (2 * 8.0**2))) * 3.0).astype("float32")
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        width=96,
        height=96,
        count=1,
        dtype="float32",
        crs="EPSG:27700",
        transform=from_origin(500000, 500096, 1, 1),
        nodata=-9999,
    ) as destination:
        destination.write(patch, 1)

    # Use a tiny descriptor dataset purely to exercise persistence/loading;
    # the production training command operates on LiDAR-derived descriptors.
    rng = np.random.default_rng(8)
    positive = rng.normal(1.0, 0.1, size=(24, 3096)).astype("float32")
    negative = rng.normal(-1.0, 0.1, size=(24, 3096)).astype("float32")
    x = np.vstack([positive, negative])
    y = np.asarray([1] * 24 + [0] * 24, dtype="uint8")
    groups = np.asarray([f"p{i//3}" for i in range(24)] + [f"n{i//3}" for i in range(24)])
    dataset = tmp_path / "training.npz"
    np.savez_compressed(dataset, X=x, y=y, groups=groups)
    from prospector.training.model import train_model
    train_model(dataset, tmp_path / "models")

    candidate = TerrainCandidate(
        1, box(500040, 500040, 500056, 500056), 70.0, 256.0, 1.2, "positive", 32.0
    )
    scored, metadata = score_candidates(tmp_path / "models" / "current.joblib", raster, [candidate])
    assert metadata["enabled"] is True
    assert len(scored) == 1
    assert 0.0 <= scored[0].trained_model_score <= 100.0
