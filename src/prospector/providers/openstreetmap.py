from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pyproj import Transformer
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import transform as transform_geometry

from prospector.providers.http import CachedResponse, HttpClient

OVERPASS_URLS = (
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
OVERPASS_URL = OVERPASS_URLS[0]


@dataclass(frozen=True, slots=True)
class OSMContextResult:
    features: list[dict[str, Any]]
    cache_entries: list[CachedResponse]
    errors: list[str]
    metadata: dict[str, Any]


def _classify(tags: dict[str, Any]) -> str | None:
    highway = str(tags.get("highway", "")).casefold()
    if tags.get("building") is not None:
        return "building"
    if tags.get("barrier") in {"fence", "wall", "hedge"} or tags.get("natural") == "hedge":
        return "boundary"
    if highway in {"footway", "path", "pedestrian", "bridleway", "cycleway", "steps"}:
        return "path"
    if highway == "track":
        return "track"
    if highway in {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service", "living_street", "road",
    }:
        return "road"
    return None


def _geometry_from_way(way: dict[str, Any]) -> Any | None:
    geometry = way.get("geometry") or []
    if len(geometry) < 2:
        return None
    points = [(float(item["lon"]), float(item["lat"])) for item in geometry if "lon" in item and "lat" in item]
    if len(points) < 2:
        return None
    if way.get("tags", {}).get("building") is not None and points[0] != points[-1]:
        points.append(points[0])
    if way.get("tags", {}).get("building") is not None and len(points) >= 4:
        return Polygon(points)
    return LineString(points)


class OpenStreetMapContextProvider:
    """Supplementary modern context for paths, tracks, fences and hedges."""

    def __init__(self, client: HttpClient) -> None:
        self.client = client

    def search(self, bbox: tuple[float, float, float, float]) -> OSMContextResult:
        xmin, ymin, xmax, ymax = bbox
        to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        west, south = to_wgs84.transform(xmin, ymin)
        east, north = to_wgs84.transform(xmax, ymax)
        query = f"""[out:json][timeout:25];
(
  way[\"highway\"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});
  way[\"barrier\"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});
  way[\"natural\"=\"hedge\"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});
  way[\"building\"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});
);
out geom;"""
        errors: list[str] = []
        cache_entries: list[CachedResponse] = []
        features: list[dict[str, Any]] = []
        payload = None
        last_error: Exception | None = None
        endpoint_used: str | None = None
        attempted_endpoints: list[str] = []
        endpoint_errors: list[str] = []
        for endpoint_index, endpoint in enumerate(OVERPASS_URLS):
            attempted_endpoints.append(endpoint)
            try:
                payload, cached = self.client.cached_json(
                    f"openstreetmap-overpass-modern-context-{endpoint_index}",
                    endpoint,
                    {"data": query},
                )
                cache_entries.append(cached)
                if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
                    raise ValueError("Overpass response did not contain an element list")
                endpoint_used = endpoint
                break
            except Exception as exc:
                last_error = exc
                endpoint_errors.append(f"{endpoint}: {exc}")
        if payload is None:
            raise RuntimeError(f"all Overpass endpoints failed: {last_error}")
        from_wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        for element in payload["elements"]:
            tags = element.get("tags") or {}
            context_type = _classify(tags)
            if context_type is None:
                continue
            geometry = _geometry_from_way(element)
            if geometry is None:
                continue
            geometry = transform_geometry(from_wgs84.transform, geometry)
            properties = {
                "source": "OpenStreetMap",
                "osm_id": element.get("id"),
                "prospector_context_type": context_type,
                **tags,
            }
            features.append({"type": "Feature", "geometry": mapping(geometry), "properties": properties})
        metadata = {
            "provider": "OpenStreetMap / Overpass",
            "endpoint": endpoint_used,
            "attempted_endpoints": attempted_endpoints,
            "endpoint_errors": endpoint_errors,
            "feature_count": len(features),
            "context_types": sorted({feature["properties"].get("prospector_context_type") for feature in features}),
        }
        return OSMContextResult(features, cache_entries, errors, metadata)
