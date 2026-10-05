from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prospector.providers.http import CachedResponse, HttpClient

WORLD_IMAGERY_EXPORT_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
)


def _validate_png(data: bytes) -> None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("World Imagery export did not return a PNG image")


@dataclass(frozen=True, slots=True)
class ImageryResult:
    preview_path: Path | None
    cache_entries: list[CachedResponse]
    errors: list[str]
    metadata: dict[str, Any]


class HighResolutionImageryProvider:
    """Download a high-resolution World Imagery image aligned to the study bbox."""

    def __init__(self, client: HttpClient) -> None:
        self.client = client

    @staticmethod
    def _output_size(
        bbox_bng: tuple[float, float, float, float],
        max_dimension: int = 2048,
    ) -> tuple[int, int, float, float]:
        xmin, ymin, xmax, ymax = bbox_bng
        width_m = max(1.0, float(xmax - xmin))
        height_m = max(1.0, float(ymax - ymin))
        aspect = width_m / height_m
        if aspect >= 1.0:
            width = max_dimension
            height = max(1, round(max_dimension / aspect))
        else:
            height = max_dimension
            width = max(1, round(max_dimension * aspect))
        return width, height, width_m / width, height_m / height

    def acquire(
        self,
        bbox_bng: tuple[float, float, float, float],
        run_dir: Path,
        *,
        max_dimension: int = 2048,
    ) -> ImageryResult:
        width, height, x_resolution, y_resolution = self._output_size(
            bbox_bng,
            max_dimension=max_dimension,
        )
        xmin, ymin, xmax, ymax = bbox_bng
        params = {
            "bbox": f"{xmin:.3f},{ymin:.3f},{xmax:.3f},{ymax:.3f}",
            "bboxSR": "27700",
            "imageSR": "27700",
            "size": f"{width},{height}",
            "format": "png32",
            "transparent": "false",
            "f": "image",
        }
        errors: list[str] = []
        cache_entries: list[CachedResponse] = []
        output_path = run_dir / "overlays" / "high-resolution-imagery.png"
        metadata: dict[str, Any] = {
            "enabled": False,
            "provider": "Esri World Imagery",
            "service": WORLD_IMAGERY_EXPORT_URL,
            "output_width_px": width,
            "output_height_px": height,
            "requested_resolution_x_m_per_px": x_resolution,
            "requested_resolution_y_m_per_px": y_resolution,
            "attribution": "Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            "description": (
                "High-resolution aerial/satellite basemap export aligned to the study bbox. "
                "World Imagery resolution varies by location and source."
            ),
        }
        try:
            cached = self.client.cached_bytes(
                "world-imagery-export",
                WORLD_IMAGERY_EXPORT_URL,
                params,
                suffix=".png",
                validator=_validate_png,
            )
            cache_entries.append(cached)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(cached.path.read_bytes())
            metadata["enabled"] = True
            metadata["cache_hit"] = cached.cache_hit
            return ImageryResult(output_path, cache_entries, errors, metadata)
        except Exception as exc:
            errors.append(f"High-resolution imagery acquisition failed: {exc}")
            return ImageryResult(None, cache_entries, errors, metadata)
