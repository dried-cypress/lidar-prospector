from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union

from prospector.providers.historic_england import HistoricEnglandProvider
from prospector.providers.http import HttpClient
from prospector.providers.lidar import LidarProvider
from prospector.training.db import ensure_schema, query_features, register_model, upsert_features
from prospector.training.features import DEFAULT_ROTATIONS, DEFAULT_SCALES_M, patch_from_raster
from prospector.training.model import train_model


@dataclass(frozen=True, slots=True)
class IngestSummary:
    tiles: int
    records: int
    errors: tuple[str, ...]


def _grid(bbox: tuple[float, float, float, float], tile_size_m: float) -> list[tuple[float, float, float, float]]:
    xmin, ymin, xmax, ymax = bbox
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be positive")
    tiles: list[tuple[float, float, float, float]] = []
    x = xmin
    while x < xmax - 1e-9:
        y = ymin
        x2 = min(x + tile_size_m, xmax)
        while y < ymax - 1e-9:
            tiles.append((x, y, x2, min(y + tile_size_m, ymax)))
            y += tile_size_m
        x += tile_size_m
    return tiles


def ingest_he_grid(
    bbox: tuple[float, float, float, float],
    *,
    tile_size_m: float,
    client: HttpClient,
    database_url: str,
    max_tiles: int | None = None,
) -> IngestSummary:
    ensure_schema(database_url)
    provider = HistoricEnglandProvider(client)
    errors: list[str] = []
    tiles = _grid(bbox, tile_size_m)
    if max_tiles is not None:
        tiles = tiles[: max(0, int(max_tiles))]
    total_records = 0
    completed = 0
    for index, tile in enumerate(tiles, 1):
        tile_wkt = box(*tile).wkt
        try:
            result = provider.search(tile_wkt)
            features = result.features + result.monument_extents + result.project_areas
            total_records += upsert_features(features, database_url)
            errors.extend(result.errors)
            completed += 1
        except Exception as exc:
            errors.append(f"HE tile {index}/{len(tiles)} {tile}: {exc}")
    return IngestSummary(completed, total_records, tuple(errors))


def _feature_centre(feature: dict[str, Any]) -> tuple[float, float]:
    geometry = shape(feature["geometry"])
    point = geometry.representative_point()
    return float(point.x), float(point.y)


def _feature_scale(feature: dict[str, Any]) -> float:
    geometry = shape(feature["geometry"])
    minx, miny, maxx, maxy = geometry.bounds
    width = max(1.0, maxx - minx)
    height = max(1.0, maxy - miny)
    equivalent = max(4.0, 2.0 * math.sqrt(max(float(geometry.area), 1.0) / math.pi))
    size = max(width, height, equivalent) * 2.5
    return float(min(256.0, max(32.0, size)))


def _sample_negative_centres(
    raster_path: Path,
    forbidden: Any,
    count: int,
    *,
    seed: int,
) -> list[tuple[float, float]]:
    import rasterio
    rng = random.Random(seed)
    with rasterio.open(raster_path) as dataset:
        bounds = dataset.bounds
        centres: list[tuple[float, float]] = []
        attempts = max(100, count * 30)
        for _ in range(attempts):
            if len(centres) >= count:
                break
            x = rng.uniform(bounds.left, bounds.right)
            y = rng.uniform(bounds.bottom, bounds.top)
            point = __import__("shapely.geometry", fromlist=["Point"]).Point(x, y)
            if forbidden is not None and forbidden.contains(point):
                continue
            centres.append((x, y))
        return centres


def _write_dataset_manifest(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    manifest = output_dir / "examples.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return manifest


def build_dataset(
    raster_tiles: list[Path],
    features_by_tile: dict[Path, list[dict[str, Any]]],
    output_dir: Path,
    *,
    negative_multiplier: int = 2,
    max_features: int | None = None,
    scales: tuple[float, ...] = DEFAULT_SCALES_M,
    rotations: tuple[float, ...] = DEFAULT_ROTATIONS,
) -> tuple[Path, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptors: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    rows: list[dict[str, Any]] = []
    rng = random.Random(20261005)
    feature_count = 0
    positive_count = 0
    negative_count = 0

    for tile_path in raster_tiles:
        features = features_by_tile.get(tile_path, [])
        if max_features is not None and feature_count >= max_features:
            break
        if max_features is not None:
            features = features[: max(0, max_features - feature_count)]
        if not features:
            continue
        forbidden = unary_union([shape(feature["geometry"]).buffer(8.0) for feature in features])
        for feature_index, feature in enumerate(features, 1):
            centre_x, centre_y = _feature_centre(feature)
            base_scale = _feature_scale(feature)
            uid = str(feature.get("source_uid") or f"feature-{feature_index}")
            for scale_factor in (0.75, 1.0, 1.5):
                scale_m = float(np.clip(base_scale * scale_factor, 32.0, 256.0))
                for rotation in rotations:
                    vector = patch_from_raster(tile_path, centre_x, centre_y, scale_m, rotation_deg=rotation)
                    descriptors.append(vector)
                    labels.append(1)
                    groups.append(f"positive:{uid}")
                    rows.append({
                        "label": 1,
                        "example_type": "he-positive",
                        "source_uid": uid,
                        "tile": str(tile_path),
                        "scale_m": scale_m,
                        "rotation_deg": rotation,
                    })
                    positive_count += 1
            feature_count += 1
        negative_target = max(1, len(features) * int(max(1, negative_multiplier)))
        negative_centres = _sample_negative_centres(tile_path, forbidden, negative_target, seed=rng.randint(0, 2**31 - 1))
        for negative_index, (x, y) in enumerate(negative_centres, 1):
            scale_m = float(rng.choice(scales))
            rotation = float(rng.choice(rotations))
            vector = patch_from_raster(tile_path, x, y, scale_m, rotation_deg=rotation)
            descriptors.append(vector)
            labels.append(0)
            groups.append(f"background:{tile_path.stem}")
            rows.append({
                "label": 0,
                "example_type": "background",
                "tile": str(tile_path),
                "scale_m": scale_m,
                "rotation_deg": rotation,
                "index": negative_index,
            })
            negative_count += 1

    if not descriptors or not any(labels) or all(labels):
        raise ValueError("Dataset preparation did not produce both archaeology and background examples")
    x = np.stack(descriptors).astype("float32")
    y = np.asarray(labels, dtype="uint8")
    group_array = np.asarray(groups, dtype="U256")
    dataset_path = output_dir / "training-dataset.npz"
    np.savez_compressed(dataset_path, X=x, y=y, groups=group_array)
    _write_dataset_manifest(output_dir, rows)
    metadata = {
        "feature_version": "patch-v1",
        "examples": int(len(y)),
        "positive_examples": int(positive_count),
        "negative_examples": int(negative_count),
        "feature_dimensions": int(x.shape[1]),
        "scales_m": list(scales),
        "rotations_deg": list(rotations),
    }
    (output_dir / "dataset.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return dataset_path, {"examples": len(y), "positive": positive_count, "negative": negative_count}
