from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prospector.providers.http import CachedResponse, HttpClient

DEFAULT_REVERSE_GEOCODER_URL = "https://nominatim.openstreetmap.org/reverse"


@dataclass(frozen=True, slots=True)
class LocationResult:
    name: str | None
    display_name: str | None
    cache_entry: CachedResponse | None
    error: str | None
    metadata: dict[str, Any]
    cache_entries: tuple[CachedResponse, ...] = ()


def _best_name(payload: dict[str, Any]) -> str | None:
    object_type = str(payload.get("type") or payload.get("category") or "").casefold()
    weak_poi_types = {
        "pub", "bar", "restaurant", "cafe", "fast_food", "shop", "supermarket",
        "building", "road", "parking", "fuel", "bus_stop", "toilets",
    }

    namedetails = payload.get("namedetails") or {}
    if isinstance(namedetails, dict) and object_type not in weak_poi_types:
        for key in ("name:en", "name"):
            value = namedetails.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    address = payload.get("address") or {}
    if isinstance(address, dict):
        for key in (
            "historic", "natural", "archaeological_site", "peak", "hill",
            "mountain", "park", "attraction", "island", "archipelago",
            "locality", "hamlet", "village", "town", "city", "amenity",
        ):
            value = address.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if isinstance(namedetails, dict):
        for key in ("name:en", "name"):
            value = namedetails.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    value = payload.get("display_name")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _object_type(payload: dict[str, Any]) -> str:
    return str(payload.get("type") or payload.get("category") or "").casefold()



class LocationProvider:
    """Resolve a coordinate to a human-readable place name.

    The provider defaults to Nominatim but can be pointed at another compatible
    reverse-geocoder via ``PROSPECTOR_GEOCODER_URL``. Results are cached through
    Prospector's normal HTTP cache, so a run makes at most one reverse-geocode
    request for a coordinate.
    """

    def __init__(self, client: HttpClient, url: str = DEFAULT_REVERSE_GEOCODER_URL) -> None:
        self.client = client
        self.url = url

    def reverse(self, latitude: float, longitude: float) -> LocationResult:
        base_params = {
            "format": "jsonv2",
            "lat": f"{latitude:.7f}",
            "lon": f"{longitude:.7f}",
            "addressdetails": 1,
            "namedetails": 1,
        }
        weak_poi_types = {
            "pub", "bar", "restaurant", "cafe", "fast_food", "shop", "supermarket",
            "building", "road", "parking", "fuel", "bus_stop", "toilets",
        }
        landscape_types = {
            "natural", "peak", "hill", "mountain", "historic",
            "archaeological_site", "park", "island", "archipelago",
        }
        cache_entries: list[CachedResponse] = []
        try:
            payload, cache_entry = self.client.cached_json(
                "reverse-geocode",
                self.url,
                {**base_params, "zoom": 18},
                validator=lambda value: self._validate(value),
                headers={"Accept": "application/json"},
            )
            cache_entries.append(cache_entry)
            name = _best_name(payload)
            object_type = _object_type(payload)
            fallback_used = False

            # Prefer landscape-scale names over nearby businesses. We probe
            # progressively broader Nominatim zooms until we find a genuinely
            # landscape/historic object. Localities are accepted only as the
            # final fallback, so a pub cannot become the report title.
            if object_type in weak_poi_types or object_type not in landscape_types:
                landscape_payload = None
                landscape_cache = None
                for zoom in (16, 14, 12, 10):
                    candidate, candidate_cache = self.client.cached_json(
                        f"reverse-geocode-landscape-{zoom}",
                        self.url,
                        {**base_params, "zoom": zoom},
                        validator=lambda value: self._validate(value),
                        headers={"Accept": "application/json"},
                    )
                    cache_entries.append(candidate_cache)
                    candidate_type = _object_type(candidate)
                    candidate_name = _best_name(candidate)
                    if candidate_name and candidate_type in landscape_types:
                        landscape_payload = candidate
                        landscape_cache = candidate_cache
                        name = candidate_name
                        object_type = candidate_type
                        fallback_used = True
                        break

                if landscape_payload is not None:
                    payload = landscape_payload
                    cache_entry = landscape_cache or cache_entry
                elif object_type in weak_poi_types:
                    # Strip the business name but retain a useful settlement or
                    # locality name when no stronger landscape object is found.
                    address = payload.get("address") or {}
                    locality = next(
                        (str(address[key]).strip() for key in ("locality", "village", "town", "city")
                         if isinstance(address, dict) and str(address.get(key) or "").strip()),
                        None,
                    )
                    if locality:
                        name = locality
                        object_type = "locality"

            return LocationResult(
                name=name,
                display_name=str(payload.get("display_name") or "") or None,
                cache_entry=cache_entry,
                error=None if name else "Reverse geocoder returned no useful place name",
                metadata={
                    "provider": "reverse geocoder",
                    "url": self.url,
                    "object_type": object_type,
                    "landscape_fallback_used": fallback_used,
                    "attribution": "© OpenStreetMap contributors",
                },
                cache_entries=tuple(cache_entries),
            )
        except Exception as exc:
            return LocationResult(
                name=None,
                display_name=None,
                cache_entry=None,
                error=str(exc),
                metadata={
                    "provider": "reverse geocoder",
                    "url": self.url,
                },
                cache_entries=tuple(cache_entries),
            )

    @staticmethod
    def _validate(payload: Any) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Reverse geocoder response was not a JSON object")


def best_heritage_feature_name(
    features: list[dict[str, Any]],
    easting: float,
    northing: float,
    *,
    max_distance_m: float = 3000.0,
) -> str | None:
    """Return the nearest useful named archaeological landscape feature.

    Historic England feature exports can carry different name fields depending
    on the source/project. Prefer explicit site/feature names and ignore pure
    identifiers or generic monument types.
    """
    from shapely.geometry import Point, shape

    point = Point(float(easting), float(northing))
    preferred_keys = (
        "NAME", "NAME_1", "NAME_2", "SITE_NAME", "MONUMENT_NAME",
        "FEATURE_NAME", "SITE", "PLACE_NAME", "LOCATION",
    )
    semantic_bonus = ("hill", "barrow", "camp", "fort", "earthwork", "enclosure", "castle", "settlement")
    best: tuple[float, str] | None = None
    for feature in features:
        properties = feature.get("properties") or {}
        geometry_data = feature.get("geometry")
        if not geometry_data:
            continue
        try:
            distance = point.distance(shape(geometry_data))
        except Exception:
            continue
        if distance > max_distance_m:
            continue
        seen_names: set[str] = set()
        candidate_values: list[tuple[int, str]] = []
        for key_index, key in enumerate(preferred_keys):
            raw = properties.get(key)
            if isinstance(raw, str) and raw.strip():
                candidate_values.append((key_index, raw.strip()))
        # HE project exports are not perfectly schema-stable. Accept other
        # name/site/location/place/monument fields when present, while avoiding
        # identifiers and generic classification values.
        for raw_key, raw_value in properties.items():
            key_lower = str(raw_key).casefold()
            if not any(token in key_lower for token in ("name", "site", "place", "location", "monument")):
                continue
            if any(token in key_lower for token in ("uid", "guid", "id", "type", "period", "evidence", "source")):
                continue
            if isinstance(raw_value, str) and raw_value.strip():
                candidate_values.append((len(preferred_keys), raw_value.strip()))

        for key_index, name in candidate_values:
            if name in seen_names:
                continue
            seen_names.add(name)
            lowered = name.casefold()
            if lowered in {"unknown", "unnamed", "-", "n/a"}:
                continue
            score = -distance + (500.0 - min(key_index, len(preferred_keys)) * 60.0)
            if any(token in lowered for token in semantic_bonus):
                score += 600.0
            if len(name) > 90:
                score -= 150.0
            if best is None or score > best[0]:
                best = (score, name)
    return best[1] if best else None
