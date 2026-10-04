from __future__ import annotations

from pathlib import Path

import pytest

from prospector.providers.lidar import DTM_COVERAGE_ID, LidarProvider
from prospector.reporting.overlay import HE_CONTEXT_STYLES


def test_lidar_coverage_id_parser_selects_expected_coverage() -> None:
    xml = (
        b'<Capabilities xmlns="http://www.opengis.net/wcs/2.0">'
        b"<Contents>"
        b"<CoverageSummary><CoverageId>other-coverage</CoverageId></CoverageSummary>"
        b"<CoverageSummary><CoverageId>"
        + DTM_COVERAGE_ID.encode()
        + b"</CoverageId></CoverageSummary>"
        b"</Contents></Capabilities>"
    )
    assert LidarProvider.coverage_id(xml) == DTM_COVERAGE_ID


def test_lidar_capabilities_validator_accepts_expected_coverage() -> None:
    xml = (
        b'<Capabilities xmlns="http://www.opengis.net/wcs/2.0">'
        b"<Contents><CoverageSummary><CoverageId>"
        + DTM_COVERAGE_ID.encode()
        + b"</CoverageId></CoverageSummary></Contents></Capabilities>"
    )
    LidarProvider.validate_capabilities(xml)


def test_lidar_capabilities_validator_rejects_missing_expected_coverage() -> None:
    with pytest.raises(ValueError, match="expected.*coverage"):
        LidarProvider.validate_capabilities(
            b'<Capabilities xmlns="http://www.opengis.net/wcs/2.0">'
            b"<Contents><CoverageSummary><CoverageId>other</CoverageId>"
            b"</CoverageSummary></Contents></Capabilities>"
        )


def test_lidar_capabilities_validator_rejects_invalid_xml() -> None:
    with pytest.raises(Exception):
        LidarProvider.validate_capabilities(b"<Capabilities>")


def test_lidar_geotiff_rejects_xml_response() -> None:
    with pytest.raises(ValueError, match="XML"):
        LidarProvider._validate_geotiff(
            b"<ExceptionReport>Something went wrong</ExceptionReport>",
            (523000, 108000, 524000, 109000),
        )


def test_lidar_geotiff_rejects_non_tiff_response() -> None:
    with pytest.raises(ValueError, match="TIFF"):
        LidarProvider._validate_geotiff(b"not a raster", (0, 0, 1, 1))


def test_lidar_geotiff_accepts_valid_bng_geotiff(tmp_path: Path) -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "valid.tif"
    data = np.arange(100, dtype="float32").reshape(10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:27700",
        transform=from_origin(523000, 108010, 1, 1),
        nodata=-9999,
    ) as destination:
        destination.write(data, 1)
    LidarProvider._validate_geotiff(path.read_bytes(), (523001, 108001, 523009, 108009))


def test_historic_england_target_layer_constants() -> None:
    from prospector.providers.historic_england import (
        DETAILED_MAPPING_LAYER_ID,
        DETAILED_MAPPING_LAYER_NAME,
    )

    assert DETAILED_MAPPING_LAYER_ID == 0
    assert DETAILED_MAPPING_LAYER_NAME == "Detailed_Mapping"


def test_he_context_layer_styles_are_distinct_from_detailed_mapping() -> None:
    from prospector.reporting.overlay import he_style
    assert he_style("Monument_Extents")["colour"] == HE_CONTEXT_STYLES["Monument_Extents"]["colour"]
    assert he_style("Project_Area")["colour"] == HE_CONTEXT_STYLES["Project_Area"]["colour"]


def test_satellite_signing_retries_http_429(monkeypatch, tmp_path: Path) -> None:
    from prospector.providers.satellite import SatelliteProvider, SIGN_URL
    from prospector.providers.http import HttpClient

    client = HttpClient(tmp_path / "cache", 5.0)
    provider = SatelliteProvider(client)

    class Response:
        def __init__(self, status_code: int, payload: dict | None = None) -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = {"Retry-After": "0"}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    responses = iter([
        Response(429),
        Response(200, {"href": "https://signed.example/test.tif"}),
    ])

    def fake_get(url, **kwargs):
        assert url == SIGN_URL
        return next(responses)

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr("prospector.providers.satellite.time.sleep", lambda _seconds: None)
    assert provider._signed_href("https://example.test/asset.tif") == "https://signed.example/test.tif"


def test_historic_england_search_paginates_and_resolves_layers_by_name(monkeypatch) -> None:
    from prospector.providers.historic_england import HistoricEnglandProvider
    from prospector.providers.http import CachedResponse

    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def cached_json(self, key, url, params, *, validator=None, headers=None):
            self.calls.append((key, url, dict(params)))
            if key == "historic-england-service-v035":
                payload = {
                    "layers": [
                        {"id": 30, "name": "Project_Area", "type": "Feature Layer"},
                        {"id": 10, "name": "Monument_Extents", "type": "Feature Layer"},
                        {"id": 20, "name": "Detailed_Mapping", "type": "Feature Layer"},
                    ]
                }
            else:
                layer_id = url.rsplit("/", 2)[-2]
                offset = int(params["resultOffset"])
                names = {
                    "20": "Detailed_Mapping",
                    "10": "Monument_Extents",
                    "30": "Project_Area",
                }
                start = offset
                all_features = [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [start, 0], [start + 0.5, 0],
                                [start + 0.5, 0.5], [start, 0.5], [start, 0]
                            ]],
                        },
                        "properties": {
                            "OBJECTID": start + 1,
                            "LAYER": names[layer_id],
                            "MONUMENT_TYPE": "bank",
                            "EVIDENCE_1": "earthwork",
                        },
                    },
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [start + 1, 0], [start + 1.5, 0],
                                [start + 1.5, 0.5], [start + 1, 0.5], [start + 1, 0]
                            ]],
                        },
                        "properties": {
                            "OBJECTID": start + 2,
                            "LAYER": names[layer_id],
                            "MONUMENT_TYPE": "ditch",
                            "EVIDENCE_1": "earthwork",
                        },
                    },
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [start + 2, 0], [start + 2.5, 0],
                                [start + 2.5, 0.5], [start + 2, 0.5], [start + 2, 0]
                            ]],
                        },
                        "properties": {
                            "OBJECTID": start + 3,
                            "LAYER": names[layer_id],
                            "MONUMENT_TYPE": "structure",
                            "EVIDENCE_1": "earthwork",
                        },
                    },
                ]
                payload = {
                    "features": all_features[offset:offset + 2],
                    "exceededTransferLimit": offset == 0,
                }
            cached = CachedResponse(__file__, False, "0" * 64)
            return payload, cached

    monkeypatch.setattr(
        "prospector.providers.historic_england.QUERY_PAGE_SIZE",
        2,
    )
    client = FakeClient()
    provider = HistoricEnglandProvider(client)
    result = provider.search("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")

    assert len(result.features) == 3
    assert len(result.monument_extents) == 3
    assert len(result.project_areas) == 3
    assert result.metadata["combined_feature_count"] == 9
    assert not result.metadata["truncated"]
    assert result.metadata["layers"]["Detailed_Mapping"]["layer_id"] == 20
    assert result.metadata["layers"]["Detailed_Mapping"]["pages_returned"] == 2
    assert sum(1 for key, _url, _params in client.calls if key.startswith("historic-england-layer-v035-20")) == 2


def test_satellite_reproject_accepts_integer_source_data() -> None:
    import numpy as np
    from rasterio.transform import from_origin
    from prospector.providers.satellite import SatelliteProvider

    source = np.arange(9, dtype="uint16").reshape(3, 3)
    result = SatelliteProvider._reproject(
        source,
        from_origin(0, 3, 1, 1),
        "EPSG:4326",
        (3, 3),
        from_origin(0, 3, 1, 1),
        "EPSG:4326",
    )
    assert result.dtype == np.float32
    assert result.shape == (3, 3)
    assert float(np.nanmax(result)) == 8.0


def test_location_provider_selects_english_name_and_caches_result(tmp_path: Path) -> None:
    from prospector.providers.http import CachedResponse
    from prospector.providers.location import LocationProvider

    class FakeClient:
        def cached_json(self, key, url, params, *, validator=None, headers=None):
            assert key == "reverse-geocode"
            assert params["format"] == "jsonv2"
            assert headers["Accept"] == "application/json"
            payload = {
                "display_name": "Thundersbarrow, West Sussex, England, United Kingdom",
                "namedetails": {"name:en": "Thundersbarrow Hill", "name": "Thundersbarrow"},
                "address": {"village": "Pyecombe"},
            }
            return payload, CachedResponse(tmp_path / "location.json", True, "0" * 64)

    result = LocationProvider(FakeClient()).reverse(50.8620, -0.2547)
    assert result.name == "Thundersbarrow Hill"
    assert result.display_name == "Thundersbarrow, West Sussex, England, United Kingdom"
    assert result.error is None
    assert result.metadata["attribution"] == "© OpenStreetMap contributors"


def test_satellite_read_window_converts_masked_uint16_before_filling(monkeypatch):
    import contextlib
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from prospector.providers.satellite import SatelliteProvider

    provider = SatelliteProvider(None)  # type: ignore[arg-type]
    transform_value = from_origin(0, 10, 1, 1)

    class FakeDataset:
        crs = "EPSG:27700"
        transform = transform_value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, *args, **kwargs):
            data = np.array([[100, 200], [300, 400]], dtype="uint16")
            mask = np.array([[False, True], [False, False]])
            return np.ma.array(data, mask=mask)

        def window_transform(self, window):
            return self.transform

    monkeypatch.setattr(provider, "_signed_href", lambda href: href)
    monkeypatch.setattr(rasterio, "open", lambda *_a, **_k: FakeDataset())
    monkeypatch.setattr(rasterio, "Env", lambda **_kwargs: contextlib.nullcontext())

    data, _transform, _crs = provider._read_asset_window(
        "scene.tif",
        (0, 8, 2, 10),
        (2, 2),
    )
    assert data.dtype == np.float32
    assert np.isnan(data[0, 1])
    assert data[1, 1] == 400.0


def test_satellite_visual_preview_uses_single_signed_asset(monkeypatch, tmp_path: Path) -> None:
    import warnings
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.transform import from_origin
    from prospector.providers.satellite import SatelliteProvider

    provider = SatelliteProvider(None)  # type: ignore[arg-type]
    calls = []

    def fake_signed(href: str) -> str:
        calls.append(href)
        return href

    monkeypatch.setattr(provider, "_signed_href", fake_signed)

    class FakeDataset:
        count = 3
        crs = "EPSG:27700"
        transform = from_origin(0, 10, 1, 1)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, indexes, window, boundless, masked):
            assert indexes == [1, 2, 3]
            import numpy as np
            from numpy.ma import masked_array
            return masked_array(np.ones((3, 10, 10), dtype="uint8"))

        def window_transform(self, window):
            return self.transform

    class FakeRasterio:
        pass

    # Exercise the method directly by replacing rasterio.open/Env in the module
    # import namespace used at runtime.
    import rasterio
    monkeypatch.setattr(rasterio, "open", lambda *_a, **_k: FakeDataset())
    monkeypatch.setattr(rasterio, "Env", lambda **_kwargs: __import__("contextlib").nullcontext())

    def fake_reproject(data, *_args, **_kwargs):
        return data

    monkeypatch.setattr(provider, "_reproject", staticmethod(fake_reproject))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        output = provider._preview_rgb(
            {"assets": {"visual": {"href": "visual.tif"}}},
            (0, 0, 5, 5),
            (10, 10),
            from_origin(0, 10, 1, 1),
            tmp_path / "preview.png",
        )
    assert output is not None
    assert calls == ["visual.tif"]
    assert output.read_bytes().startswith(b"\x89PNG")

