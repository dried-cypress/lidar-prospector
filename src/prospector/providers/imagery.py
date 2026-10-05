from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prospector.providers.http import CachedResponse, HttpClient

WORLD_IMAGERY_EXPORT_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
)


def _validate_png(data: bytes) -> None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("World Imagery export did not return a PNG image")


def _map_content_rect(
    bbox: tuple[float, float, float, float],
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox
    map_aspect = max(1e-9, (xmax - xmin) / max(1e-9, ymax - ymin))
    canvas_aspect = canvas_width / canvas_height
    if map_aspect >= canvas_aspect:
        width = canvas_width
        height = max(1, round(canvas_width / map_aspect))
        left = 0
        top = (canvas_height - height) // 2
    else:
        height = canvas_height
        width = max(1, round(canvas_height * map_aspect))
        left = (canvas_width - width) // 2
        top = 0
    return left, top, width, height


@dataclass(frozen=True, slots=True)
class ImageryResult:
    preview_path: Path | None
    cache_entries: list[CachedResponse]
    errors: list[str]
    metadata: dict[str, Any]


class HighResolutionImageryProvider:
    """Acquire World Imagery and snap it to the exact LiDAR display grid."""

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

    @staticmethod
    def _write_source_geotiff(
        path: Path,
        png_path: Path,
        bbox_bng: tuple[float, float, float, float],
    ) -> Path:
        import rasterio
        from rasterio.transform import from_bounds
        from PIL import Image

        with Image.open(png_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype="uint8")
        height, width, _ = rgb.shape
        transform = from_bounds(*bbox_bng, width=width, height=height)

        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs="EPSG:27700",
            transform=transform,
            compress="deflate",
            photometric="RGB",
        ) as destination:
            destination.write(np.transpose(rgb, (2, 0, 1)))
        return path

    @staticmethod
    def _align_to_dtm(
        source_path: Path,
        dtm_path: Path,
        destination_path: Path,
    ) -> Path:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(source_path) as source, rasterio.open(dtm_path) as reference:
            profile = reference.profile.copy()
            profile.update(
                driver="GTiff",
                count=3,
                dtype="uint8",
                nodata=None,
                compress="deflate",
                photometric="RGB",
            )
            with rasterio.open(destination_path, "w", **profile) as destination:
                for band in range(1, 4):
                    output = np.zeros((reference.height, reference.width), dtype="uint8")
                    reproject(
                        source=rasterio.band(source, band),
                        destination=output,
                        src_transform=source.transform,
                        src_crs=source.crs,
                        dst_transform=reference.transform,
                        dst_crs=reference.crs,
                        dst_nodata=0,
                        resampling=Resampling.bilinear,
                    )
                    destination.write(output, band)
        return destination_path

    @staticmethod
    def _render_report_canvas(
        aligned_path: Path,
        output_path: Path,
        bbox_bng: tuple[float, float, float, float],
        *,
        canvas_width: int = 1800,
        canvas_height: int = 1350,
    ) -> Path:
        import rasterio
        from PIL import Image

        with rasterio.open(aligned_path) as dataset:
            rgb = np.transpose(dataset.read([1, 2, 3]), (1, 2, 0))
            image = Image.fromarray(rgb.astype("uint8"), mode="RGB")

        left, top, width, height = _map_content_rect(
            bbox_bng,
            canvas_width,
            canvas_height,
        )
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        canvas.paste(image, (left, top))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, format="PNG", optimize=True)
        return output_path

    def acquire(
        self,
        bbox_bng: tuple[float, float, float, float],
        run_dir: Path,
        *,
        reference_raster: Path | None = None,
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
            # ArcGIS otherwise adjusts the requested bbox to fit the image size.
            # That changes the geographic extent and is exactly the kind of
            # horizontal/vertical overhang that breaks our LiDAR/HE alignment.
            "adjustAspectRatio": "false",
            "f": "image",
        }
        errors: list[str] = []
        cache_entries: list[CachedResponse] = []
        raw_output = run_dir / "terrain" / "high-resolution-imagery-source.png"
        aligned_output = run_dir / "terrain" / "high-resolution-imagery-aligned.tif"
        output_path = run_dir / "overlays" / "high-resolution-imagery.png"
        metadata: dict[str, Any] = {
            "enabled": False,
            "provider": "Esri World Imagery",
            "service": WORLD_IMAGERY_EXPORT_URL,
            "output_width_px": width,
            "output_height_px": height,
            "requested_resolution_x_m_per_px": x_resolution,
            "requested_resolution_y_m_per_px": y_resolution,
            "canvas_width_px": 1800,
            "canvas_height_px": 1350,
            "attribution": "Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            "description": (
                "High-resolution World Imagery acquired in British National Grid, "
                "then reprojected onto the exact LiDAR raster grid for overlay alignment."
            ),
        }
        try:
            cached = self.client.cached_bytes(
                "world-imagery-export-v044",
                WORLD_IMAGERY_EXPORT_URL,
                params,
                suffix=".png",
                validator=_validate_png,
            )
            cache_entries.append(cached)
            raw_output.parent.mkdir(parents=True, exist_ok=True)
            raw_output.write_bytes(cached.path.read_bytes())

            if reference_raster is not None and reference_raster.is_file():
                source_geotiff = run_dir / "terrain" / "high-resolution-imagery-source.tif"
                self._write_source_geotiff(source_geotiff, raw_output, bbox_bng)
                self._align_to_dtm(source_geotiff, reference_raster, aligned_output)
                self._render_report_canvas(aligned_output, output_path, bbox_bng)
                metadata.update({
                    "enabled": True,
                    "cache_hit": cached.cache_hit,
                    "georeferenced_source": str(source_geotiff),
                    "aligned_raster": str(aligned_output),
                    "alignment": "exact LiDAR raster grid",
                })
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(raw_output.read_bytes())
                metadata.update({
                    "enabled": True,
                    "cache_hit": cached.cache_hit,
                    "alignment": "requested BNG bbox",
                })

            return ImageryResult(output_path, cache_entries, errors, metadata)
        except Exception as exc:
            errors.append(f"High-resolution imagery acquisition failed: {exc}")
            return ImageryResult(None, cache_entries, errors, metadata)
