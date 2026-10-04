from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pyproj import Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry

from prospector.providers.http import CachedResponse, HttpClient

OS_DOWNLOADS_PRODUCTS_URL = "https://api.os.uk/downloads/v1/products"
OS_OPENMAP_LOCAL_NAME = "OS OpenMap - Local"

# 100 km BNG square lettering, ordered south-to-north and west-to-east.  The
# table is deliberately explicit because this is also easier to audit than a
# magic-letter arithmetic formula.
_BNG_100KM = (
    ("SV", "SW", "SX", "SY", "SZ", "TV", "TW"),
    ("SQ", "SR", "SS", "ST", "SU", "TQ", "TR"),
    ("SL", "SM", "SN", "SO", "SP", "TL", "TM"),
    ("SF", "SG", "SH", "SJ", "SK", "TF", "TG"),
    ("SA", "SB", "SC", "SD", "SE", "TA", "TB"),
    ("NV", "NW", "NX", "NY", "NZ", "OV", "OW"),
    ("NQ", "NR", "NS", "NT", "NU", "OQ", "OR"),
    ("NL", "NM", "NN", "NO", "NP", "OL", "OM"),
    ("NF", "NG", "NH", "NJ", "NK", "OF", "OG"),
    ("NA", "NB", "NC", "ND", "NE", "OA", "OB"),
    ("HV", "HW", "HX", "HY", "HZ", "JV", "JW"),
    ("HQ", "HR", "HS", "HT", "HU", "JQ", "JR"),
    ("HL", "HM", "HN", "HO", "HP", "JL", "JM"),
)


@dataclass(frozen=True, slots=True)
class OSContextResult:
    features: list[dict[str, Any]]
    cache_entries: list[CachedResponse]
    errors: list[str]
    metadata: dict[str, Any]


def _normalise(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").replace("-", " ").split())


def _iter_named_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _iter_named_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_named_dicts(item)


def bng_100km_square(easting: float, northing: float) -> str:
    e = int(easting // 100_000)
    n = int(northing // 100_000)
    if n < 0 or n >= len(_BNG_100KM) or e < 0 or e >= len(_BNG_100KM[n]):
        raise ValueError(f"Coordinate {easting:.1f}, {northing:.1f} is outside supported BNG tiles")
    return _BNG_100KM[n][e]


def bng_squares_for_bounds(bounds: tuple[float, float, float, float]) -> list[str]:
    xmin, ymin, xmax, ymax = bounds
    # epsilon keeps a coordinate exactly on the northern/eastern edge in the
    # lower tile rather than inventing a second tile outside the requested area.
    eps = 1e-6
    corners = ((xmin, ymin), (xmax - eps, ymin), (xmin, ymax - eps), (xmax - eps, ymax - eps))
    squares = {bng_100km_square(x, y) for x, y in corners}
    return sorted(squares)


def _classify_feature(layer_name: str, properties: dict[str, Any]) -> str | None:
    text = _normalise(layer_name + " " + " ".join(f"{k} {v}" for k, v in properties.items()))
    if "building" in text:
        return "building"
    if any(token in text for token in ("footpath", "bridleway", "cycleway", "pedestrian", "path")):
        return "path"
    if "track" in text:
        return "track"
    if any(token in text for token in ("road", "motorway", "trunk", "street", "carriageway")):
        return "road"
    if any(token in text for token in ("fence", "hedge", "wall", "boundary")):
        return "boundary"
    return None


def _extract_download_urls(payload: Any) -> list[str]:
    urls: list[str] = []
    for item in _iter_named_dicts(payload):
        for key, value in item.items():
            if not isinstance(value, str) or not value.startswith(("http://", "https://")):
                continue
            key_norm = _normalise(key)
            if any(token in key_norm for token in ("url", "href", "download")) or value.lower().endswith(".zip"):
                if value not in urls:
                    urls.append(value)
    return [url for url in urls if ".zip" in url.lower() or "download" in url.lower()]


def _find_product(payload: Any) -> dict[str, Any] | None:
    target = _normalise(OS_OPENMAP_LOCAL_NAME)
    for item in _iter_named_dicts(payload):
        name = _normalise(item.get("name"))
        if name == target or ("openmap" in name and "local" in name):
            if "id" in item:
                return item
    return None


def _feature_crs(feature_source: Any) -> str | None:
    crs = getattr(feature_source, "crs", None)
    if isinstance(crs, dict):
        init = crs.get("init")
        if isinstance(init, str):
            return init
        name = crs.get("name")
        if isinstance(name, str):
            return name
    return str(crs) if crs else None



OS_FEATURES_WFS_URL = "https://api.os.uk/features/v1/wfs"
OS_FEATURES_LAYERS: tuple[tuple[str, str], ...] = (
    ("Zoomstack_LocalBuildings", "building"),
    ("Zoomstack_DistrictBuildings", "building"),
    ("Zoomstack_RoadsLocal", "road"),
    ("Zoomstack_RoadsRegional", "road"),
    ("Zoomstack_RoadsNational", "road"),
    ("Zoomstack_Rail", "road"),
    ("Zoomstack_Waterlines", "water"),
)


class OSFeaturesProvider:
    """Fast, AOI-scoped OS context using the OS Features API / Zoomstack.

    Unlike the OS OpenMap Local download, this requests only the features in the
    analysis bbox. It requires an OS API key and is therefore preferred whenever
    one is configured. The OpenMap Local downloader remains available as an
    explicit/offline path.
    """

    def __init__(self, client: HttpClient, api_key: str) -> None:
        self.client = client
        self.api_key = api_key

    def _layer_features(
        self,
        type_name: str,
        context_type: str,
        bbox: tuple[float, float, float, float],
    ) -> tuple[list[dict[str, Any]], list[CachedResponse]]:
        xmin, ymin, xmax, ymax = bbox
        features: list[dict[str, Any]] = []
        cache_entries: list[CachedResponse] = []
        start_index = 0
        page = 0
        while True:
            params = {
                "request": "GetFeature",
                "service": "WFS",
                "version": "2.0.0",
                "typeNames": type_name,
                # OS documents bbox order as bottom-left y, bottom-left x,
                # top-right y, top-right x.
                "bbox": f"{ymin:.3f},{xmin:.3f},{ymax:.3f},{xmax:.3f}",
                "srsName": "EPSG:27700",
                "outputFormat": "GEOJSON",
                "count": 100,
                "startIndex": start_index,
            }
            payload, cached = self.client.cached_json(
                f"os-features-{type_name}-{page}",
                OS_FEATURES_WFS_URL,
                params,
                headers={"key": self.api_key},
            )
            cache_entries.append(cached)
            page_features = payload.get("features", []) if isinstance(payload, dict) else []
            if not isinstance(page_features, list):
                raise ValueError(f"OS Features API response for {type_name} did not contain features")
            for feature in page_features:
                if not isinstance(feature, dict) or not feature.get("geometry"):
                    continue
                props = dict(feature.get("properties") or {})
                props.update(
                    {
                        "source": "Ordnance Survey OS Features API / OS Open Zoomstack",
                        "source_layer": type_name,
                        "prospector_context_type": context_type,
                    }
                )
                features.append(
                    {
                        "type": "Feature",
                        "geometry": feature["geometry"],
                        "properties": props,
                    }
                )
            page += 1
            if len(page_features) < 100 or page >= 20:
                break
            start_index += len(page_features)
        return features, cache_entries

    def search(self, bbox: tuple[float, float, float, float]) -> OSContextResult:
        features: list[dict[str, Any]] = []
        cache_entries: list[CachedResponse] = []
        errors: list[str] = []
        try:
            # Keep the number of transactions low and deterministic. The rural
            # 1 km default AOI normally completes with one page per layer.
            for type_name, context_type in OS_FEATURES_LAYERS:
                layer_features, layer_cache = self._layer_features(type_name, context_type, bbox)
                features.extend(layer_features)
                cache_entries.extend(layer_cache)
        except Exception as exc:
            errors.append(f"OS Features API acquisition failed: {exc}")
        return OSContextResult(
            features=features,
            cache_entries=cache_entries,
            errors=errors,
            metadata={
                "provider": "Ordnance Survey OS Features API / OS Open Zoomstack",
                "feature_count": len(features),
                "context_types": sorted({feature["properties"].get("prospector_context_type") for feature in features}),
                "layers": [layer for layer, _ in OS_FEATURES_LAYERS],
                "api_key_configured": bool(self.api_key),
            },
        )


class OSOpenMapProvider:
    """Fetch OS OpenMap Local context for buildings and transport features."""

    def __init__(self, client: HttpClient) -> None:
        self.client = client

    def _discover_product(self) -> tuple[dict[str, Any], CachedResponse]:
        payload, cached = self.client.cached_json(
            "os-openmap-products",
            OS_DOWNLOADS_PRODUCTS_URL,
            {},
        )
        product = _find_product(payload)
        if product is None:
            raise RuntimeError("OS Downloads API did not expose an OS OpenMap - Local product")
        return product, cached

    def _download_tile(self, product_id: str, square: str) -> tuple[CachedResponse, dict[str, Any]]:
        url = f"{OS_DOWNLOADS_PRODUCTS_URL}/{product_id}/downloads"
        last_error: Exception | None = None
        for fmt in ("GeoPackage", "ESRI Shapefile", "GML"):
            try:
                payload, metadata_cache = self.client.cached_json(
                    f"os-openmap-downloads-{product_id}-{square}-{fmt}",
                    url,
                    {"format": fmt, "area": square},
                )
                urls = _extract_download_urls(payload)
                if not urls:
                    continue
                file_url = urls[0]
                package_cache = self.client.cached_bytes(
                    f"os-openmap-tile-{square}-{fmt}",
                    file_url,
                    {},
                    suffix=".zip",
                )
                return package_cache, {"format": fmt, "metadata_cache": metadata_cache, "download_url": file_url}
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not download OS OpenMap Local tile {square}: {last_error}")

    @staticmethod
    def _read_vector_sources(extracted_dir: Path, bbox: tuple[float, float, float, float]) -> Iterable[tuple[str, Any, dict[str, Any]]]:
        try:
            import fiona
        except ImportError as exc:
            raise RuntimeError("OS vector processing requires fiona; install with: pip install -e '.[geo]'") from exc

        vector_files = [
            path for path in extracted_dir.rglob("*")
            if path.suffix.casefold() in {".gpkg", ".shp", ".gml"} and path.is_file()
        ]
        seen_layers: set[tuple[str, str]] = set()
        for path in sorted(vector_files):
            if path.suffix.casefold() == ".gpkg":
                layers = list(fiona.listlayers(path))
            else:
                layers = [None]
            for layer in layers:
                with fiona.open(path, layer=layer, bbox=bbox) as source:
                    source_crs = _feature_crs(source)
                    transformer = None
                    if source_crs and "27700" not in source_crs:
                        transformer = Transformer.from_crs(source_crs, "EPSG:27700", always_xy=True)
                    layer_key = (str(path), str(layer))
                    if layer_key in seen_layers:
                        continue
                    seen_layers.add(layer_key)
                    for feature in source:
                        geometry_data = feature.get("geometry")
                        if not geometry_data:
                            continue
                        try:
                            geometry = shape(geometry_data)
                            if transformer is not None:
                                geometry = transform_geometry(transformer.transform, geometry)
                            geometry = geometry.intersection(box(*bbox))
                        except Exception:
                            continue
                        if geometry.is_empty:
                            continue
                        properties = dict(feature.get("properties") or {})
                        context_type = _classify_feature(str(layer or path.stem), properties)
                        if context_type is None:
                            continue
                        properties.update(
                            {
                                "source": "Ordnance Survey OS OpenMap Local",
                                "source_layer": str(layer or path.stem),
                                "prospector_context_type": context_type,
                            }
                        )
                        yield context_type, geometry, properties

    def search(
        self,
        bbox: tuple[float, float, float, float],
        output_dir: Path,
        local_data: Path | None = None,
    ) -> OSContextResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        features: list[dict[str, Any]] = []
        cache_entries: list[CachedResponse] = []
        errors: list[str] = []
        package_sources: list[dict[str, Any]] = []
        seen: set[tuple[str, bytes]] = set()

        try:
            if local_data is not None:
                package_paths = [local_data]
                squares: list[str] = []
                product = {"id": "local", "name": "Local OS OpenMap data"}
            else:
                product, product_cache = self._discover_product()
                cache_entries.append(product_cache)
                squares = bng_squares_for_bounds(bbox)
                package_paths = []
                for square in squares:
                    package_cache, package_meta = self._download_tile(str(product["id"]), square)
                    cache_entries.extend([package_cache, package_meta["metadata_cache"]])
                    package_path = output_dir / square / package_cache.path.name
                    package_path.parent.mkdir(parents=True, exist_ok=True)
                    package_path.write_bytes(package_cache.path.read_bytes())
                    package_paths.append(package_path)
                    package_sources.append({"square": square, "format": package_meta["format"], "download_url": package_meta["download_url"]})

            for package in package_paths:
                extracted = package.parent / package.stem
                if package.is_dir():
                    extracted = package
                elif not extracted.exists():
                    extracted.mkdir(parents=True, exist_ok=True)
                    if zipfile.is_zipfile(package):
                        with zipfile.ZipFile(package) as archive:
                            archive.extractall(extracted)
                    else:
                        # A local shapefile/GPKG may be supplied directly.
                        extracted = package.parent

                for context_type, geometry, properties in self._read_vector_sources(extracted, bbox):
                    dedupe_key = (context_type, geometry.wkb)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    features.append({"type": "Feature", "geometry": mapping(geometry), "properties": properties})
        except Exception as exc:
            errors.append(f"OS OpenMap Local acquisition failed: {exc}")

        metadata = {
            "provider": "Ordnance Survey OS OpenMap Local",
            "feature_count": len(features),
            "context_types": sorted({feature["properties"].get("prospector_context_type") for feature in features}),
            "package_sources": package_sources,
            "local_data": str(local_data) if local_data else None,
        }
        return OSContextResult(features, cache_entries, errors, metadata)
