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


def _best_name(payload: dict[str, Any]) -> str | None:
    namedetails = payload.get("namedetails") or {}
    if isinstance(namedetails, dict):
        for key in ("name:en", "name"):
            value = namedetails.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    address = payload.get("address") or {}
    if isinstance(address, dict):
        for key in (
            "archipelago",
            "island",
            "park",
            "attraction",
            "historic",
            "natural",
            "amenity",
            "locality",
            "hamlet",
            "village",
            "town",
            "city",
        ):
            value = address.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    value = payload.get("display_name")
    return value.strip() if isinstance(value, str) and value.strip() else None


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
        params = {
            "format": "jsonv2",
            "lat": f"{latitude:.7f}",
            "lon": f"{longitude:.7f}",
            "zoom": 18,
            "addressdetails": 1,
            "namedetails": 1,
        }
        try:
            payload, cache_entry = self.client.cached_json(
                "reverse-geocode",
                self.url,
                params,
                validator=lambda value: self._validate(value),
                headers={"Accept": "application/json"},
            )
            name = _best_name(payload)
            return LocationResult(
                name=name,
                display_name=str(payload.get("display_name") or "") or None,
                cache_entry=cache_entry,
                error=None if name else "Reverse geocoder returned no useful place name",
                metadata={
                    "provider": "reverse geocoder",
                    "url": self.url,
                    "attribution": "© OpenStreetMap contributors",
                },
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
            )

    @staticmethod
    def _validate(payload: Any) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Reverse geocoder response was not a JSON object")
