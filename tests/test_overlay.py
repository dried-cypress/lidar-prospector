from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

from prospector.reporting.overlay import (
    HE_CONTEXT_STYLES,
    HE_LAYER_STYLES,
    create_lidar_base_overlay,
    create_lidar_aim_overlay,
    create_lidar_anomaly_aim_overlay,
    create_lidar_anomaly_overlay,
    he_style,
)
from prospector.terrain.anomalies import TerrainCandidate
from shapely.geometry import Polygon


def _write_test_raster(path: Path) -> None:
    data = np.arange(100, dtype="float32").reshape(10, 10)
    import rasterio

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:27700",
        transform=from_origin(100010, 100010, 1, 1),
        nodata=-9999,
    ) as dataset:
        dataset.write(data, 1)


def test_lidar_aim_overlay_uses_raster_extent_and_uppercase_he_layer(tmp_path: Path) -> None:
    dtm = tmp_path / "dtm.tif"
    output = tmp_path / "overlay.png"
    _write_test_raster(dtm)

    create_lidar_aim_overlay(
        dtm,
        [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[100009, 100001], [100015, 100001], [100015, 100008], [100009, 100008], [100009, 100001]]]},
                "properties": {"LAYER": "BANK", "MONUMENT_TYPE": "Bank"},
            }
        ],
        "POLYGON ((100011 100001, 100019 100001, 100019 100009, 100011 100009, 100011 100001))",
        100015,
        100005,
        output,
        render_bounds=(100010, 100000, 100020, 100010),
        monument_extents=[{
            "type":"Feature",
            "geometry":{"type":"Polygon","coordinates":[[[100011,100001],[100019,100001],[100019,100009],[100011,100009],[100011,100001]]]},
            "properties":{"prospector_aim_layer":"Monument_Extents"},
        }],
        project_areas=[{
            "type":"Feature",
            "geometry":{"type":"Polygon","coordinates":[[[100010,100000],[100020,100000],[100020,100010],[100010,100010],[100010,100000]]]},
            "properties":{"prospector_aim_layer":"Project_Area"},
        }],
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_he_style_matches_live_historic_england_layer_ids() -> None:
    assert he_style("BANK")["colour"] == "#A50026"
    assert he_style("DITCH")["colour"] == "#313695"
    assert he_style("EXTENT_OF_FEATURE")["colour"] == "#FDAE61"
    assert he_style("RIDGE_AND_FURROW_ALIGNMENT")["colour"] == "#74ADD1"
    assert he_style("RIDGE_AND_FURROW_AREA")["colour"] == "#74ADD1"
    assert he_style("SCARP_SLOPE_EDGE")["colour"] == "#4575B4"
    assert he_style("STRUCTURE")["colour"] == "#F46D43"


def test_unknown_he_layer_is_not_the_old_purple_fallback() -> None:
    assert he_style("SOMETHING_NEW") ["colour"] != "#8A2BE2"


def test_archaeological_hillshade_has_usable_contrast() -> None:
    from prospector.terrain.anomalies import calculate_archaeological_hillshade

    x = np.arange(100, dtype="float32").reshape(10, 10)
    shade = calculate_archaeological_hillshade(x, 1.0, 1.0)
    assert shade.dtype == np.uint8
    assert int(shade.max()) - int(shade.min()) > 50


def test_combined_anomaly_aim_overlay_is_rendered(tmp_path: Path) -> None:
    dtm = tmp_path / "dtm.tif"
    output = tmp_path / "combined.png"
    _write_test_raster(dtm)
    candidate = TerrainCandidate(
        1,
        Polygon([(100012, 100002), (100015, 100002), (100015, 100005), (100012, 100005), (100012, 100002)]),
        2.0,
        20.0,
        1.0,
        "positive",
        16.0,
    )
    create_lidar_anomaly_aim_overlay(
        dtm,
        [candidate],
        [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[100013, 100003], [100017, 100003], [100017, 100007], [100013, 100007], [100013, 100003]]]},
            "properties": {"LAYER": "STRUCTURE", "HE_UID": "2"},
        }],
        output,
    )
    assert output.is_file()
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_lidar_anomaly_overlay_without_he_context_renders(tmp_path: Path) -> None:
    from PIL import Image

    dtm = tmp_path / "dtm.tif"
    output = tmp_path / "anomaly.png"
    _write_test_raster(dtm)
    candidate = TerrainCandidate(
        1,
        Polygon([(100012, 100002), (100015, 100002), (100015, 100005), (100012, 100005), (100012, 100002)]),
        2.0,
        20.0,
        1.0,
        "positive",
        16.0,
    )
    create_lidar_anomaly_overlay(dtm, [candidate], output)
    assert output.is_file()
    assert Image.open(output).size == (1800, 1350)


def test_combined_overlay_accepts_all_historic_england_layers(tmp_path: Path) -> None:
    dtm = tmp_path / "dtm.tif"
    output = tmp_path / "combined.png"
    _write_test_raster(dtm)
    candidate = TerrainCandidate(
        1,
        Polygon([(100012, 100002), (100015, 100002), (100015, 100005), (100012, 100005), (100012, 100002)]),
        2.0,
        20.0,
        1.0,
        "positive",
        16.0,
    )
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[100013, 100003], [100017, 100003], [100017, 100007], [100013, 100007], [100013, 100003]]]},
        "properties": {"LAYER": "BANK", "HE_UID": "2"},
    }
    context = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[100014, 100004], [100016, 100004], [100016, 100006], [100014, 100006], [100014, 100004]]]},
        "properties": {"LAYER": "Monument_Extents", "HE_UID": "3"},
    }
    project = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[100011, 100001], [100018, 100001], [100018, 100008], [100011, 100008], [100011, 100001]]]},
        "properties": {"LAYER": "Project_Area", "HE_UID": "4"},
    }
    create_lidar_anomaly_aim_overlay(
        dtm, [candidate], [feature], output,
        monument_extents=[context], project_areas=[project],
    )
    assert output.is_file()
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"



def test_lidar_aim_png_contains_all_primary_historic_england_colours(tmp_path: Path) -> None:
    from PIL import Image

    dtm = tmp_path / "dtm.tif"
    output = tmp_path / "lidar-aim.png"
    _write_test_raster(dtm)
    features = []
    for index, layer in enumerate(("BANK", "DITCH", "SCARP_SLOPE_EDGE", "STRUCTURE")):
        y = 100001.0 + index * 2.0
        if layer == "STRUCTURE":
            geometry = {
                "type": "Polygon",
                "coordinates": [[[100011, y], [100016, y], [100016, y + 1], [100011, y + 1], [100011, y]]],
            }
        else:
            geometry = {"type": "LineString", "coordinates": [[100011, y], [100019, y]]}
        features.append({"type": "Feature", "geometry": geometry, "properties": {"LAYER": layer}})

    create_lidar_aim_overlay(
        dtm,
        features,
        "POLYGON ((100010 100000,100020 100000,100020 100010,100010 100010,100010 100000))",
        100015,
        100005,
        output,
        monument_extents=[],
        project_areas=[],
    )

    pixels = np.asarray(Image.open(output).convert("RGB"), dtype=float)
    for layer in ("Bank", "Ditch", "Slope", "Structure"):
        colour = np.asarray([
            int(HE_LAYER_STYLES[layer]["colour"][i:i + 2], 16)
            for i in (1, 3, 5)
        ], dtype=float)
        distance = np.sqrt(((pixels - colour) ** 2).sum(axis=2))
        assert int((distance < 90.0).sum()) > 20, layer

def test_combined_png_rasterises_monument_and_detail_layers(tmp_path: Path) -> None:
    from PIL import Image

    dtm = tmp_path / "dtm.tif"
    output = tmp_path / "combined.png"
    _write_test_raster(dtm)
    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[100001, 100004], [100009, 100004]]},
        "properties": {"LAYER": "DITCH"},
    }
    extent = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[100001, 100001], [100009, 100001], [100009, 100009], [100001, 100009], [100001, 100001]]]},
        "properties": {"LAYER": "Monument_Extents"},
    }
    project = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[100000, 100000], [100010, 100000], [100010, 100010], [100000, 100010], [100000, 100000]]]},
        "properties": {"LAYER": "Project_Area"},
    }
    create_lidar_anomaly_aim_overlay(
        dtm,
        [],
        [feature],
        output,
        monument_extents=[extent],
        project_areas=[project],
    )
    image = Image.open(output)
    assert image.size == (1800, 1350)
    assert image.getbbox() is not None


def test_lidar_base_overlay_is_rendered_for_report_layer_switching(tmp_path: Path) -> None:
    from PIL import Image

    dtm = tmp_path / "dtm.tif"
    output = tmp_path / "lidar-base.png"
    _write_test_raster(dtm)
    create_lidar_base_overlay(dtm, output)
    assert Image.open(output).size == (1800, 1350)


def test_lidar_aim_png_repairs_invalid_he_geometry_instead_of_dropping_it(tmp_path: Path) -> None:
    from PIL import Image

    dtm = tmp_path / "dtm.tif"
    output = tmp_path / "invalid-geometry.png"
    _write_test_raster(dtm)
    # Self-intersecting bow-tie polygon: Shapely reports this as invalid, but
    # Historic England geometry should not disappear from the PNG just because
    # direct clipping would otherwise fail.
    invalid_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[100011, 100001], [100019, 100009], [100019, 100001], [100011, 100009], [100011, 100001]]],
        },
        "properties": {"LAYER": "BANK"},
    }
    create_lidar_aim_overlay(
        dtm,
        [invalid_feature],
        "POLYGON ((100010 100000,100020 100000,100020 100010,100010 100010,100010 100000))",
        100015,
        100005,
        output,
        monument_extents=[],
        project_areas=[],
    )
    pixels = np.asarray(Image.open(output).convert("RGB"), dtype=float)
    bank = np.asarray([165.0, 0.0, 38.0])
    distance = np.sqrt(((pixels - bank) ** 2).sum(axis=2))
    assert int((distance < 100.0).sum()) > 20
