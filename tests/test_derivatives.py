from __future__ import annotations

from pathlib import Path

import numpy as np

from prospector.terrain.derivatives import calculate_hillshade


def test_calculate_hillshade_returns_uint8() -> None:
    elevation = np.arange(100, dtype="float64").reshape(10, 10)
    result = calculate_hillshade(elevation, 1.0, 1.0)
    assert result.shape == elevation.shape
    assert result.dtype == np.uint8
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


def test_multidirectional_hillshade_returns_uint8() -> None:
    from prospector.terrain.anomalies import calculate_multidirectional_hillshade

    elevation = np.arange(121, dtype="float32").reshape(11, 11)
    result = calculate_multidirectional_hillshade(elevation, 1.0, 1.0)
    assert result.dtype == np.uint8
    assert result.shape == elevation.shape


def test_detect_terrain_anomalies_returns_candidates(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from prospector.terrain.anomalies import detect_terrain_anomalies

    yy, xx = np.mgrid[0:100, 0:100]
    mound = 8.0 * np.exp(-(((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 5.0**2)))
    data = mound.astype("float32")
    dtm = tmp_path / "mound.tif"
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
        destination.write(data, 1)

    candidates, metadata = detect_terrain_anomalies(dtm, [])
    assert metadata["name"] == "multi-scale-local-relief"
    assert candidates
    assert all(candidate.area_m2 >= 12.0 for candidate in candidates)


def test_monument_extent_is_excluded_from_discovery(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from prospector.terrain.anomalies import detect_terrain_anomalies

    yy, xx = np.mgrid[0:300, 0:300]
    data = (0.02 * yy).astype("float32")
    data[:, 145:148] += 1.5
    dtm = tmp_path / "monument-exclusion.tif"
    with rasterio.open(
        dtm, "w", driver="GTiff", width=300, height=300, count=1,
        dtype="float32", crs="EPSG:27700", transform=from_origin(500000, 500300, 1, 1), nodata=-9999,
    ) as destination:
        destination.write(data, 1)

    # The monument occupies the middle portion of the line; candidate discovery
    # must remove that registered envelope while retaining the outside portions.
    extent = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[
            [500120, 500080], [500180, 500080], [500180, 500220],
            [500120, 500220], [500120, 500080],
        ]]},
        "properties": {"LAYER": "Monument_Extents"},
    }
    candidates, metadata = detect_terrain_anomalies(dtm, [], monument_extents=[extent])
    assert candidates
    assert metadata["historic_england_reference"]["monument_extents_exclusion"] is True
    assert metadata["historic_england_reference"]["candidate_generation_requires_historic_england"] is False

    for candidate in candidates:
        assert not candidate.geometry.within(extent_geometry := __import__("shapely.geometry", fromlist=["shape"]).shape(extent["geometry"]))


def test_registered_monument_contains_no_discovery_candidates(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from prospector.terrain.anomalies import detect_terrain_anomalies

    yy, xx = np.mgrid[0:160, 0:160]
    ring = ((xx - 80) ** 2 + (yy - 80) ** 2)
    data = np.exp(-((ring - 28**2) ** 2) / (2 * 6**2)).astype("float32")
    dtm = tmp_path / "registered-monument.tif"
    with rasterio.open(
        dtm, "w", driver="GTiff", width=160, height=160, count=1,
        dtype="float32", crs="EPSG:27700", transform=from_origin(610000, 610160, 1, 1), nodata=-9999,
    ) as destination:
        destination.write(data, 1)

    extent = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[609990, 609990], [610170, 609990], [610170, 610170], [609990, 610170], [609990, 609990]]]},
        "properties": {"LAYER": "Monument_Extents"},
    }
    candidates, _metadata = detect_terrain_anomalies(dtm, [], monument_extents=[extent])
    assert candidates == []


def test_detailed_mapping_does_not_create_trace_halos(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from prospector.terrain.anomalies import detect_terrain_anomalies

    yy, xx = np.mgrid[0:220, 0:220]
    data = (0.01 * yy).astype("float32")
    data[105:108, :] += 1.0
    dtm = tmp_path / "detailed-exclusion.tif"
    with rasterio.open(
        dtm, "w", driver="GTiff", width=220, height=220, count=1,
        dtype="float32", crs="EPSG:27700", transform=from_origin(600000, 600220, 1, 1), nodata=-9999,
    ) as destination:
        destination.write(data, 1)

    known = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[600000, 600110], [600220, 600110]]},
        "properties": {"LAYER": "Bank", "MONUMENT_TYPE": "Bank"},
    }
    candidates, metadata = detect_terrain_anomalies(dtm, [known])
    assert metadata["historic_england_reference"]["detailed_mapping_exclusion_buffer_m"] > 0
    known_line = __import__("shapely.geometry", fromlist=["shape"]).shape(known["geometry"])
    assert all(not candidate.geometry.intersects(known_line) for candidate in candidates)
    assert metadata["provisional_statistics"]["known_outline_traces_rejected"] >= 0



def test_crossing_unmapped_linear_feature_survives_known_feature_mask(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import LineString
    from prospector.terrain.anomalies import detect_terrain_anomalies

    yy, xx = np.mgrid[0:300, 0:300]
    data = (0.005 * yy).astype("float32")
    data[:, 145:148] += 1.2  # N-S prospect.
    data[145:148, :] += 0.8  # E-W known bank crossing it.
    dtm = tmp_path / "crossing-earthwork.tif"
    with rasterio.open(
        dtm, "w", driver="GTiff", width=300, height=300, count=1,
        dtype="float32", crs="EPSG:27700", transform=from_origin(700000, 700300, 1, 1), nodata=-9999,
    ) as destination:
        destination.write(data, 1)

    known = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[700000, 700153], [700300, 700153]]},
        "properties": {"LAYER": "Bank", "MONUMENT_TYPE": "Bank"},
    }
    candidates, _metadata = detect_terrain_anomalies(dtm, [known])
    assert candidates
    known_line = LineString(known["geometry"]["coordinates"])
    # At least one retained candidate must be linear and must not simply be the
    # mapped E-W feature; crossing candidates can be split at the known line.
    assert any(candidate.linear_score >= 50.0 and not candidate.geometry.intersects(known_line) for candidate in candidates)


def test_long_straight_earthwork_gets_linear_priority(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from prospector.terrain.anomalies import detect_terrain_anomalies

    yy, xx = np.mgrid[0:300, 0:300]
    data = (0.015 * yy).astype("float32")
    data[:, 145:148] += 1.25
    dtm = tmp_path / "straight-earthwork.tif"
    with rasterio.open(
        dtm,
        "w",
        driver="GTiff",
        width=300,
        height=300,
        count=1,
        dtype="float32",
        crs="EPSG:27700",
        transform=from_origin(510000, 510300, 1, 1),
        nodata=-9999,
    ) as destination:
        destination.write(data, 1)

    candidates, metadata = detect_terrain_anomalies(dtm, [])
    assert candidates
    linear = max(candidates, key=lambda candidate: candidate.linear_score)
    morphology = dict(linear.morphology)
    assert linear.linear_score >= 50.0
    assert morphology["elongation"] >= 10.0
    assert "very elongated linear morphology" in linear.reasons
    assert metadata["straight_line_seeded"] is True


def test_sensitivity_profiles_expose_progressively_broader_detection() -> None:
    from prospector.terrain.anomalies import SENSITIVITY_PROFILES

    low = SENSITIVITY_PROFILES["low"]
    medium = SENSITIVITY_PROFILES["medium"]
    high = SENSITIVITY_PROFILES["high"]
    assert low.threshold_percentile > medium.threshold_percentile > high.threshold_percentile
    assert low.min_area_m2 > medium.min_area_m2 > high.min_area_m2
    assert low.max_candidates < medium.max_candidates < high.max_candidates
    assert low.min_line_length_m > medium.min_line_length_m > high.min_line_length_m


def test_detector_accepts_explicit_worker_count(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from prospector.terrain.anomalies import detect_terrain_anomalies

    yy, xx = np.mgrid[0:120, 0:120]
    data = (0.01 * yy + 2.0 * np.exp(-(((xx - 60) ** 2 + (yy - 60) ** 2) / 120.0))).astype("float32")
    dtm = tmp_path / "workers.tif"
    with rasterio.open(
        dtm, "w", driver="GTiff", width=120, height=120, count=1,
        dtype="float32", crs="EPSG:27700", transform=from_origin(800000, 800120, 1, 1), nodata=-9999,
    ) as destination:
        destination.write(data, 1)
    _candidates, metadata = detect_terrain_anomalies(dtm, [], sensitivity="low", workers=1)
    assert metadata["sensitivity"] == "low"
    assert metadata["workers_requested"] == 1
