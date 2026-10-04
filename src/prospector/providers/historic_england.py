from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from shapely import wkt
from shapely.geometry import shape

from prospector.providers.http import CachedResponse, HttpClient

AIM_SERVICE = (
    "https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/arcgis/rest/services/"
    "HE_AIM_data/FeatureServer"
)
AIM_EXPERIENCE_URL = "https://experience.arcgis.com/experience/3aad11a5fd384d66b0d7078448f629d2"

# Historic England's public FeatureServer advertises a 2,000-record limit.
# Keep the page size at the service maximum and explicitly page every layer so
# an AOI can never silently lose features merely because the first response is
# full.
QUERY_PAGE_SIZE = 2000
MAX_QUERY_PAGES = 100

# Retained for backwards compatibility with callers/tests which imported the
# original IDs. Runtime layer selection is by published layer name because the
# service metadata is the authoritative source and IDs may change.
DETAILED_MAPPING_LAYER_ID = 0
DETAILED_MAPPING_LAYER_NAME = "Detailed_Mapping"
MONUMENT_EXTENTS_LAYER_ID = 1
MONUMENT_EXTENTS_LAYER_NAME = "Monument_Extents"
PROJECT_AREA_LAYER_ID = 2
PROJECT_AREA_LAYER_NAME = "Project_Area"


@dataclass(frozen=True, slots=True)
class HistoricEnglandSearchResult:
    # `features` remains the detailed-mapping list for anomaly scoring/backwards compatibility.
    features: list[dict[str, Any]]
    monument_extents: list[dict[str, Any]]
    project_areas: list[dict[str, Any]]
    cache_entries: list[CachedResponse]
    errors: list[str]
    metadata: dict[str, Any]


class HistoricEnglandProvider:
    """Adapter for Historic England's public Aerial Investigation and Mapping service."""

    def __init__(self, client: HttpClient) -> None:
        self.client = client

    @staticmethod
    def _validate_service(payload: Any) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Historic England service response was not a JSON object")
        if "error" in payload:
            raise ValueError(f"Historic England service returned an error: {payload['error']}")
        if not isinstance(payload.get("layers", []), list):
            raise ValueError("Historic England service response has invalid layers")

    @staticmethod
    def _validate_query(payload: Any) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Historic England query response was not a JSON object")
        if "error" in payload:
            raise ValueError(f"Historic England query returned an error: {payload['error']}")
        if not isinstance(payload.get("features"), list):
            raise ValueError("Historic England query response has no feature list")

    @staticmethod
    def _normalise_layer_name(value: object) -> str:
        return " ".join(
            str(value or "").strip().casefold().replace("_", " ").replace("-", " ").split()
        )

    def discover_layers(self) -> tuple[list[dict[str, Any]], CachedResponse]:
        # Versioned cache key intentionally prevents an older cached service
        # definition from pinning runtime layer IDs after a service change.
        payload, cached = self.client.cached_json(
            "historic-england-service-v035",
            AIM_SERVICE,
            {"f": "json"},
            validator=self._validate_service,
        )
        layers = payload.get("layers", []) + payload.get("tables", [])
        return [layer for layer in layers if isinstance(layer, dict)], cached

    def _query_layer(
        self,
        layer: dict[str, Any],
        geometry: Any,
        geometry_json: dict[str, Any],
        cache_entries: list[CachedResponse],
        errors: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        layer_id = str(layer["id"])
        layer_name = str(layer["name"])
        url = f"{AIM_SERVICE}/{layer_id}/query"
        base_params = {
            "f": "geojson",
            "where": "1=1",
            "geometry": json.dumps(geometry_json, separators=(",", ":"), sort_keys=True),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 27700,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 27700,
        }

        features: list[dict[str, Any]] = []
        page = 0
        offset = 0
        exceeded_transfer_limit = False
        query_failed = False

        while page < MAX_QUERY_PAGES:
            params = {
                **base_params,
                "resultOffset": offset,
                "resultRecordCount": QUERY_PAGE_SIZE,
            }
            try:
                payload, cached = self.client.cached_json(
                    f"historic-england-layer-v035-{layer_id}-page-{page}",
                    url,
                    params,
                    validator=self._validate_query,
                )
            except Exception as exc:
                errors.append(f"{layer_name}: page {page + 1}: {exc}")
                query_failed = True
                break

            cache_entries.append(cached)
            page += 1
            page_features = payload.get("features", [])
            exceeded = bool(payload.get("exceededTransferLimit", False))
            exceeded_transfer_limit = exceeded_transfer_limit or exceeded

            for feature in page_features:
                geometry_data = feature.get("geometry")
                if geometry_data is None:
                    continue
                try:
                    feature_geometry = shape(geometry_data)
                except Exception as exc:
                    errors.append(f"{layer_name}: invalid feature geometry: {exc}")
                    continue
                if not feature_geometry.intersects(geometry):
                    continue
                original_properties = dict(feature.get("properties") or {})
                feature["properties"] = {
                    "source": "Historic England AIM",
                    "layer": layer_name,
                    "prospector_aim_layer": layer_name,
                    **original_properties,
                }
                features.append(feature)

            # A short page is the normal end condition only when the service says
            # the transfer limit has not been exceeded. Some ArcGIS services can
            # still report exceededTransferLimit on a short page, so keep paging in
            # that case. Continue after every full page as well.
            if not page_features:
                break
            if not exceeded and len(page_features) < QUERY_PAGE_SIZE:
                break
            offset += len(page_features)

        else:
            errors.append(
                f"{layer_name}: query exceeded the safety limit of "
                f"{MAX_QUERY_PAGES} pages"
            )
            exceeded_transfer_limit = True

        metadata = {
            "layer_id": int(layer_id),
            "layer_name": layer_name,
            "query_page_size": QUERY_PAGE_SIZE,
            "pages_returned": page,
            "records_returned": len(features),
            "exceeded_transfer_limit": exceeded_transfer_limit,
            "truncated": query_failed or page >= MAX_QUERY_PAGES,
        }
        return features, metadata

    def search(self, geometry_wkt: str) -> HistoricEnglandSearchResult:
        geometry = wkt.loads(geometry_wkt)
        xmin, ymin, xmax, ymax = geometry.bounds
        geometry_json = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "spatialReference": {"wkid": 27700},
        }
        detailed: list[dict[str, Any]] = []
        monument_extents: list[dict[str, Any]] = []
        project_areas: list[dict[str, Any]] = []
        cache_entries: list[CachedResponse] = []
        errors: list[str] = []
        layer_metadata: dict[str, dict[str, Any]] = {}

        layers, service_cache = self.discover_layers()
        cache_entries.append(service_cache)

        wanted = {
            DETAILED_MAPPING_LAYER_NAME: "detailed",
            MONUMENT_EXTENTS_LAYER_NAME: "monument",
            PROJECT_AREA_LAYER_NAME: "project",
        }
        found: dict[str, dict[str, Any]] = {}
        compatibility_ids = {
            DETAILED_MAPPING_LAYER_NAME: DETAILED_MAPPING_LAYER_ID,
            MONUMENT_EXTENTS_LAYER_NAME: MONUMENT_EXTENTS_LAYER_ID,
            PROJECT_AREA_LAYER_NAME: PROJECT_AREA_LAYER_ID,
        }
        for layer in layers:
            if layer.get("type") != "Feature Layer":
                continue
            layer_name = str(layer.get("name", ""))
            normalised = self._normalise_layer_name(layer_name)
            for wanted_name in wanted:
                if normalised == self._normalise_layer_name(wanted_name):
                    found[wanted_name] = layer

        # Graceful compatibility fallback if a service response omits names but
        # retains the historical numeric IDs. Name matching always wins.
        if len(found) < len(wanted):
            for layer in layers:
                if layer.get("type") != "Feature Layer":
                    continue
                try:
                    layer_id = int(layer.get("id"))
                except (TypeError, ValueError):
                    continue
                for wanted_name, compatibility_id in compatibility_ids.items():
                    if wanted_name not in found and layer_id == compatibility_id:
                        found[wanted_name] = layer

        for layer_name, kind in wanted.items():
            layer = found.get(layer_name)
            if layer is None:
                errors.append(
                    f"Historic England service does not expose the expected "
                    f"{layer_name} layer"
                )
                continue
            features, metadata = self._query_layer(
                layer, geometry, geometry_json, cache_entries, errors
            )
            layer_metadata[layer_name] = metadata
            if kind == "detailed":
                detailed = features
            elif kind == "monument":
                monument_extents = features
            else:
                project_areas = features

        truncated_layers = [
            name for name, metadata in layer_metadata.items() if metadata.get("truncated")
        ]
        raster_projects: list[dict[str, Any]] = []
        for feature in project_areas:
            properties = feature.get("properties") or {}
            mapping_type = str(properties.get("TYPE") or "").strip()
            if "raster" not in mapping_type.casefold():
                continue
            raster_projects.append({
                "project_name": properties.get("PROJECT_NA"),
                "type": mapping_type,
                "draw_format": properties.get("DRAWFORMAT"),
            })

        metadata = {
            "service_url": AIM_SERVICE,
            "explorer_url": AIM_EXPERIENCE_URL,
            "source_type": "underlying ArcGIS FeatureServer used by the Aerial Archaeology Mapping Explorer",
            "layer_resolution": "published layer name with historical-ID compatibility fallback",
            "query_page_size": QUERY_PAGE_SIZE,
            "max_query_pages": MAX_QUERY_PAGES,
            "layers": layer_metadata,
            "combined_feature_count": len(detailed) + len(monument_extents) + len(project_areas),
            "truncated": bool(truncated_layers),
            "truncated_layers": truncated_layers,
            "raster_only_project_areas": raster_projects,
            "raster_only_project_count": len(raster_projects),
        }

        return HistoricEnglandSearchResult(
            detailed, monument_extents, project_areas, cache_entries, errors, metadata
        )
