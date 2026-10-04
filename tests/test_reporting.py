from pathlib import Path

from shapely.geometry import Polygon

from prospector.reporting.html import _candidate_svg, _feature_svg, _map_viewport, write_html_report
from prospector.terrain.anomalies import TerrainCandidate


def _png(path: Path, width: int = 100, height: int = 80) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    try:
        ax.imshow([[0, 1], [1, 0]], extent=(0, width, 0, height), origin="upper", cmap="gray")
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(path, bbox_inches=None, pad_inches=0)
    finally:
        plt.close(fig)


def test_html_report_contains_descriptive_he_metadata_and_links(tmp_path: Path) -> None:
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
        "properties": {
            "LAYER": "Bank",
            "HE_UID": "123456",
            "MONUMENT_TYPE": "Bank",
            "PERIOD": "Iron Age",
            "EVIDENCE_1": "Earthwork",
            "SOURCE_1": "LiDAR",
            "HE_URL1": "https://example.test/heritage/123456",
        },
    }
    out = tmp_path / "report.html"
    write_html_report(
        out,
        run_id="test",
        application_version="0.2.1",
        latitude=50.8,
        longitude=-0.2,
        easting=530000,
        northing=105000,
        diameter_m=1000,
        aim_features=[feature],
        monument_extents=[],
        project_areas=[],
        cache_entries=[],
        errors=[],
        overlay_path=None,
        dtm_path=None,
        hillshade_path=None,
        aim_geojson_path=None,
        study_bounds=(529500, 104500, 530500, 105500),
    )
    text = out.read_text(encoding="utf-8")
    assert "Iron Age" in text
    assert "Earthwork" in text
    assert "https://example.test/heritage/123456" in text
    assert "Heritage Gateway record" in text
    assert "Combined terrain anomalies + Historic England mapping" in text


def test_html_report_groups_repeated_geometry_by_heritage_record(tmp_path: Path) -> None:
    base = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
        "properties": {
            "LAYER": "Bank",
            "HE_UID": "123456",
            "MONUMENT_TYPE": "Bank",
            "PERIOD": "Iron Age",
            "EVIDENCE_1": "Earthwork",
            "SOURCE_1": "LiDAR",
            "HE_URL1": "https://example.test/heritage/123456",
        },
    }
    second = dict(base)
    second["geometry"] = {"type": "Polygon", "coordinates": [[[20, 20], [30, 20], [30, 30], [20, 30], [20, 20]]]}
    out = tmp_path / "report.html"
    write_html_report(
        out,
        run_id="test",
        application_version="0.2.1",
        latitude=50.8,
        longitude=-0.2,
        easting=530000,
        northing=105000,
        diameter_m=1000,
        aim_features=[base, second],
        monument_extents=[],
        project_areas=[],
        cache_entries=[],
        errors=[],
        overlay_path=None,
        dtm_path=None,
        hillshade_path=None,
        aim_geojson_path=None,
        study_bounds=(0, 0, 100, 100),
    )
    text = out.read_text(encoding="utf-8")
    assert "0 project area(s), 0 monument extent(s), and 2 detailed mapped feature geometries" in text


def test_svg_feature_and_candidate_paths_are_clipped_to_image_bounds() -> None:
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[-10, -10], [110, -10], [110, 110], [-10, 110], [-10, -10]]]},
        "properties": {"LAYER": "BANK", "HE_UID": "1"},
    }
    svg = _feature_svg([feature], (0, 0, 100, 100), 100, 80)
    import re
    coords = [float(value) for value in re.findall(r"(?:^|[ ,])(-?\d+(?:\.\d+)?)", re.search(r'd="([^"]+)"', svg).group(1))]
    assert min(coords) >= 0
    assert max(coords) <= 100

    candidate = TerrainCandidate(1, Polygon([(-10, -10), (110, -10), (110, 110), (-10, 110), (-10, -10)]), 5.0, 100.0, 1.0, "positive", 16.0)
    candidate_svg = _candidate_svg([candidate], (0, 0, 100, 100), 100, 80)
    assert 'id="map-candidate-1"' in candidate_svg
    assert 'data-candidate-map-link="1"' in candidate_svg
    assert 'Candidate 1' in candidate_svg
    assert 'cx="50.00"' in candidate_svg


def test_html_uses_actual_png_dimensions_for_svg_viewbox(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.png"
    _png(overlay, 101, 77)
    report = tmp_path / "report.html"
    write_html_report(
        report,
        run_id="test",
        application_version="0.2.1",
        latitude=50.8,
        longitude=-0.2,
        easting=530000,
        northing=105000,
        diameter_m=1000,
        aim_features=[],
        monument_extents=[],
        project_areas=[],
        cache_entries=[],
        errors=[],
        overlay_path=overlay,
        dtm_path=None,
        hillshade_path=None,
        aim_geojson_path=None,
        study_bounds=(0, 0, 100, 100),
        map_bounds=(0, 0, 101, 77),
    )
    text = report.read_text(encoding="utf-8")
    assert 'viewBox="0 0 101 77"' in text



def test_html_map_viewport_matches_equal_aspect_matplotlib_canvas() -> None:
    left, top, width_pct, height_pct, viewport_width, viewport_height = _map_viewport(1800, 1350, (0, 0, 1000, 1000))
    assert round(left, 2) == 12.5
    assert round(top, 2) == 0.0
    assert round(width_pct, 2) == 75.0
    assert round(height_pct, 2) == 100.0
    assert viewport_width == 1350
    assert viewport_height == 1350


def test_html_report_positions_svg_over_equal_aspect_map_not_full_png(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.png"
    from PIL import Image
    Image.new("RGB", (1800, 1350), "white").save(overlay)
    report = tmp_path / "report.html"
    write_html_report(
        report,
        run_id="test",
        application_version="0.3.3",
        latitude=50.8,
        longitude=-0.2,
        easting=530000,
        northing=105000,
        diameter_m=1000,
        aim_features=[{
            "type":"Feature",
            "geometry":{"type":"Polygon","coordinates":[[[0,0],[1000,0],[1000,1000],[0,1000],[0,0]]]},
            "properties":{"LAYER":"BANK","HE_UID":"1"},
        }],
        monument_extents=[],
        project_areas=[],
        cache_entries=[],
        errors=[],
        overlay_path=overlay,
        dtm_path=None,
        hillshade_path=None,
        aim_geojson_path=None,
        study_bounds=(0,0,1000,1000),
        map_bounds=(0,0,1000,1000),
    )
    text = report.read_text(encoding="utf-8")
    assert 'left:12.500000%;top:0.000000%;width:75.000000%;height:100.000000%;' in text
    assert 'viewBox="0 0 1350 1350"' in text

def test_html_report_contains_context_aware_candidate_evidence(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    candidate = TerrainCandidate(
        1,
        Polygon([(0, 0), (12, 0), (12, 12), (0, 12), (0, 0)]),
        78.5,
        144.0,
        1.5,
        "positive",
        16.0,
        lidar_score=88.0,
        persistence_score=72.0,
        morphology_score=81.0,
        modern_penalty=12.0,
        satellite_support=34.0,
        he_similarity=69.0,
        classification="high-priority",
        reasons=("coherent compact / enclosure-like morphology", "persists across multiple LiDAR scales"),
        morphology=(("circularity", 0.8), ("rectangularity", 0.75)),
    )
    write_html_report(
        out,
        run_id="test",
        application_version="0.3.0",
        latitude=50.8,
        longitude=-0.2,
        easting=530000,
        northing=105000,
        diameter_m=1000,
        aim_features=[],
        monument_extents=[],
        project_areas=[],
        cache_entries=[],
        errors=[],
        overlay_path=None,
        dtm_path=None,
        hillshade_path=None,
        aim_geojson_path=None,
        study_bounds=(0, 0, 100, 100),
        candidates=[candidate],
        modern_context_path=None,
        modern_context_raster_path=None,
        satellite_preview_path=None,
        satellite_support_path=None,
        satellite_search_path=None,
    )
    report = out.read_text(encoding="utf-8")
    assert "78.5" in report
    assert "coherent compact / enclosure-like morphology" in report
    assert "Satellite" in report
    assert "HE similarity" in report



def test_html_report_has_switchable_local_evidence_bases(tmp_path: Path) -> None:
    from PIL import Image

    report = tmp_path / "report.html"
    overlay = tmp_path / "overlay.png"
    lidar_base = tmp_path / "lidar-base.png"
    anomaly_base = tmp_path / "anomaly-base.png"
    satellite = tmp_path / "satellite.png"
    for path in (overlay, lidar_base, anomaly_base, satellite):
        Image.new("RGB", (1800, 1350), "white").save(path)

    write_html_report(
        report,
        run_id="switching",
        application_version="0.3.7",
        latitude=50.8,
        longitude=-0.2,
        easting=530000,
        northing=105000,
        diameter_m=1000,
        aim_features=[],
        monument_extents=[],
        project_areas=[],
        cache_entries=[],
        errors=[],
        overlay_path=overlay,
        dtm_path=None,
        hillshade_path=None,
        aim_geojson_path=None,
        study_bounds=(0, 0, 1000, 1000),
        map_bounds=(0, 0, 1000, 1000),
        lidar_base_path=lidar_base,
        anomaly_base_path=anomaly_base,
        satellite_preview_path=satellite,
    )
    text = report.read_text(encoding="utf-8")
    assert 'data-base-button="lidar"' in text
    assert 'data-base-button="anomalies"' in text
    assert 'data-base-button="satellite"' in text
    assert "layer-map" in text
    assert "data-toggle-overlay=\"he\"" in text


def test_html_report_has_candidate_navigation_and_theme_controls(tmp_path: Path) -> None:
    report = tmp_path / "report.html"
    candidate = TerrainCandidate(1, Polygon([(10, 10), (30, 10), (30, 20), (10, 20), (10, 10)]), 88.0, 200.0, 1.2, "positive", 16.0)
    write_html_report(
        report, run_id="theme", application_version="0.3.7", latitude=50.8, longitude=-0.2,
        easting=530000, northing=105000, diameter_m=1000, aim_features=[], monument_extents=[],
        project_areas=[], cache_entries=[], errors=[], overlay_path=None, dtm_path=None,
        hillshade_path=None, aim_geojson_path=None, study_bounds=(0, 0, 100, 100),
        candidates=[candidate], anomaly_metadata={"sensitivity": "medium", "workers_requested": 0},
    )
    text = report.read_text(encoding="utf-8")
    assert 'data-theme-toggle' in text
    assert 'data-focus-candidate="1"' in text
    assert 'id="candidate-1"' in text
    assert 'map-candidate-1' in text
    assert 'Toggle dark mode' in text


def test_html_report_candidate_table_links_to_visible_map_marker(tmp_path: Path) -> None:
    from PIL import Image

    report = tmp_path / "report.html"
    base = tmp_path / "overlay.png"
    Image.new("RGB", (1000, 1000), "white").save(base)
    candidate = TerrainCandidate(1, Polygon([(10, 10), (30, 10), (30, 20), (10, 20), (10, 10)]), 88.0, 200.0, 1.2, "positive", 16.0)
    write_html_report(
        report, run_id="candidate-map", application_version="0.3.7", latitude=50.8, longitude=-0.2,
        easting=530000, northing=105000, diameter_m=1000, aim_features=[], monument_extents=[],
        project_areas=[], cache_entries=[], errors=[], overlay_path=base, dtm_path=None,
        hillshade_path=None, aim_geojson_path=None, study_bounds=(0, 0, 100, 100),
        map_bounds=(0, 0, 100, 100), candidates=[candidate],
        anomaly_metadata={"sensitivity": "medium", "workers_requested": 0},
        lidar_base_path=base,
    )
    text = report.read_text(encoding="utf-8")
    assert 'id="map-candidate-1"' in text
    assert 'data-focus-candidate="1"' in text
    assert 'View on map ↗' in text
