from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from prospector.providers.openstreetmap import _classify
from prospector.providers.os import OSOpenMapProvider


class _DummyHttpClient:
    def __init__(self) -> None:
        self.session = None



def test_bng_100km_square_and_bounds() -> None:
    from prospector.providers.os import bng_100km_square, bng_squares_for_bounds

    assert bng_100km_square(523915.55, 108829.39) == "TQ"
    assert bng_squares_for_bounds((500000, 100000, 600000, 200000)) == ["TQ"]


def test_modern_context_score_is_high_near_mapped_road(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import LineString, mapping

    from prospector.terrain.context import build_modern_context_raster

    dtm = tmp_path / "dtm.tif"
    with rasterio.open(
        dtm,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=1,
        dtype="float32",
        crs="EPSG:27700",
        transform=from_origin(500000, 500100, 1, 1),
        nodata=-9999,
    ) as destination:
        destination.write(np.zeros((100, 100), dtype="float32"), 1)

    feature = {
        "type": "Feature",
        "geometry": mapping(LineString([(500000, 500050), (500100, 500050)])),
        "properties": {"prospector_context_type": "road"},
    }
    result = build_modern_context_raster(dtm, [feature], tmp_path / "modern.tif")
    with rasterio.open(result.score_path) as dataset:
        data = dataset.read(1)
    assert float(data[50, 50]) > 0.9
    assert float(data[0, 0]) < 0.01


def test_satellite_scene_selection_spreads_dates() -> None:
    from prospector.providers.satellite import SatelliteProvider

    scenes = []
    base = datetime(2026, 9, 1, tzinfo=UTC)
    for index in range(12):
        scenes.append(
            {
                "id": f"scene-{index}",
                "properties": {
                    "datetime": (base - timedelta(days=index * 20)).isoformat(),
                    "eo:cloud_cover": 5.0,
                },
            }
        )
    selected = SatelliteProvider(None)._select_scenes(scenes, max_scenes=4)  # type: ignore[arg-type]
    assert len(selected) == 4
    dates = [datetime.fromisoformat(item["properties"]["datetime"]) for item in selected]
    assert all(abs((dates[index] - dates[index - 1]).days) >= 45 for index in range(1, len(dates)))


def test_os_local_geopackage_features_are_normalised(tmp_path: Path) -> None:
    import fiona
    from shapely.geometry import mapping

    gpkg = tmp_path / "os.gpkg"
    schema = {"geometry": "LineString", "properties": {"name": "str"}}
    with fiona.open(
        gpkg,
        mode="w",
        driver="GPKG",
        layer="Roads",
        schema=schema,
        crs="EPSG:27700",
    ) as destination:
        destination.write({
            "geometry": mapping(LineString([(500000, 100000), (500050, 100000)])),
            "properties": {"name": "Road"},
        })

    result = OSOpenMapProvider(_DummyHttpClient()).search(
        (499900, 99900, 500100, 100100), tmp_path / "output", local_data=gpkg
    )
    assert not result.errors
    assert result.features
    assert result.features[0]["properties"]["prospector_context_type"] == "road"


def test_osm_classifies_modern_feature_types() -> None:
    assert _classify({"highway": "footway"}) == "path"
    assert _classify({"highway": "track"}) == "track"
    assert _classify({"barrier": "fence"}) == "boundary"
    assert _classify({"building": "yes"}) == "building"


class _FakeOSSession:
    def __init__(self):
        self.calls = []
    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers, timeout))
        import json
        from types import SimpleNamespace
        payload = {"type": "FeatureCollection", "features": []}
        response = SimpleNamespace(
            content=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            raise_for_status=lambda: None,
        )
        return response


def test_os_features_provider_is_aoi_scoped_and_does_not_download_tiles(tmp_path: Path) -> None:
    from prospector.providers.http import HttpClient
    from prospector.providers.os import OSFeaturesProvider

    client = HttpClient(tmp_path)
    fake = _FakeOSSession()
    client.session = fake
    result = OSFeaturesProvider(client, "test-key").search((522426, 107894, 523426, 108894))
    assert not result.errors
    assert len(fake.calls) == 7
    assert all(call[2]["key"] == "test-key" for call in fake.calls)
    assert all("bbox" in call[1] for call in fake.calls)
