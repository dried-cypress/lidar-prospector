from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Point


@dataclass(frozen=True, slots=True)
class Coordinate:
    latitude: float
    longitude: float

    def validate(self) -> None:
        if not -90 <= self.latitude <= 90:
            raise ValueError("latitude must be between -90 and 90")
        if not -180 <= self.longitude <= 180:
            raise ValueError("longitude must be between -180 and 180")


@dataclass(frozen=True, slots=True)
class StudyArea:
    centre: Coordinate
    diameter_m: float
    easting: float
    northing: float
    geometry_wkt: str
    crs: str = "EPSG:27700"

    @classmethod
    def from_coordinate(cls, centre: Coordinate, diameter_m: float) -> "StudyArea":
        centre.validate()
        if diameter_m <= 0:
            raise ValueError("diameter must be greater than zero")
        if diameter_m > 10000:
            raise ValueError("diameter must not exceed 10,000 m in this development release")
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        easting, northing = transformer.transform(centre.longitude, centre.latitude)
        geometry = Point(easting, northing).buffer(diameter_m / 2.0, quad_segs=32)
        return cls(centre, diameter_m, easting, northing, geometry.wkt)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return the single EPSG:27700 bounding box used by all providers."""
        return tuple(float(value) for value in wkt.loads(self.geometry_wkt).bounds)


@dataclass(frozen=True, slots=True)
class Run:
    run_id: str
    area: StudyArea
    started_at: datetime
    application_version: str
    path: str
    run_uuid: UUID = field(default_factory=uuid4)

    @classmethod
    def create(cls, area: StudyArea, application_version: str, path: str) -> "Run":
        now = datetime.now(UTC)
        run_id = now.strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"
        return cls(run_id, area, now, application_version, path)
