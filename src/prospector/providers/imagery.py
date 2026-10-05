from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any

import numpy as np

from prospector.providers.http import CachedResponse, HttpClient

OAM_GLOBAL_MOSAIC_URL = "https://global.imagery.hotosm.org/{z}/{x}/{y}.png"
OAM_ATTRIBUTION = "OpenAerialMap / Humanitarian OpenStreetMap Team contributors"
WEB_MERCATOR_HALF = 20037508.342789244
TILE_SIZE = 256
DEFAULT_ZOOM = 17


def _validate_png(data: bytes) -> None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("OpenAerialMap tile did not return a PNG image")


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


def _web_mercator_bounds(bbox_bng: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    transform = Transformer.from_crs("EPSG:27700", "EPSG:3857", always_xy=True)
    xmin, ymin, xmax, ymax = bbox_bng
    points = [transform.transform(xmin, ymin), transform.transform(xmin, ymax), transform.transform(xmax, ymin), transform.transform(xmax, ymax)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _tile_xy(value: float, *, axis: str, zoom: int) -> int:
    n = 2**zoom
    if axis == "x":
        fraction = (value + WEB_MERCATOR_HALF) / (2.0 * WEB_MERCATOR_HALF)
    else:
        fraction = (WEB_MERCATOR_HALF - value) / (2.0 * WEB_MERCATOR_HALF)
    return int(np.clip(floor(fraction * n), 0, n - 1))


def _render_canvas(
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

    map_aspect = max(1e-9, (bbox_bng[2] - bbox_bng[0]) / max(1e-9, bbox_bng[3] - bbox_bng[1]))
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
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(image, (left, top))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


@dataclass(frozen=True, slots=True)
class ImageryResult:
    preview_path: Path | None
    cache_entries: list[CachedResponse]
    errors: list[str]
    metadata: dict[str, Any]


class HighResolutionImageryProvider:
    """Acquire high-resolution OpenAerialMap imagery and align it to the LiDAR grid."""

    def __init__(self, client: HttpClient, *, zoom: int = DEFAULT_ZOOM, workers: int = 6) -> None:
        self.client = client
        self.zoom = int(zoom)
        self.workers = max(1, int(workers))

    @staticmethod
    def _align_to_dtm(source_path: Path, dtm_path: Path, destination_path: Path) -> Path:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(source_path) as source, rasterio.open(dtm_path) as reference:
            profile = reference.profile.copy()
            profile.update(driver="GTiff", count=3, dtype="uint8", nodata=None, compress="deflate", photometric="RGB")
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

    def _download_tile(self, z: int, x: int, y: int) -> tuple[tuple[int, int], CachedResponse]:
        url = OAM_GLOBAL_MOSAIC_URL.format(z=z, x=x, y=y)
        cached = self.client.cached_bytes(
            "open-aerial-map-tile",
            url,
            {"z": z, "x": x, "y": y},
            suffix=".png",
            validator=_validate_png,
        )
        return (x, y), cached

    def acquire(
        self,
        bbox_bng: tuple[float, float, float, float],
        run_dir: Path,
        *,
        reference_raster: Path | None = None,
    ) -> ImageryResult:
        _, _, requested_x_m, requested_y_m = _output_size(bbox_bng)
        xmin, ymin, xmax, ymax = _web_mercator_bounds(bbox_bng)
        z = self.zoom
        min_x = _tile_xy(xmin, axis="x", zoom=z)
        max_x = _tile_xy(xmax, axis="x", zoom=z)
        min_y = _tile_xy(ymax, axis="y", zoom=z)
        max_y = _tile_xy(ymin, axis="y", zoom=z)
        tiles = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]
        errors: list[str] = []
        cache_entries: list[CachedResponse] = []
        failed_tiles = 0
        output_path = run_dir / "overlays" / "high-resolution-imagery.png"
        source_path = run_dir / "terrain" / "high-resolution-imagery-source.tif"
        aligned_path = run_dir / "terrain" / "high-resolution-imagery-aligned.tif"
        metadata: dict[str, Any] = {
            "enabled": False,
            "provider": "OpenAerialMap",
            "service": OAM_GLOBAL_MOSAIC_URL,
            "zoom": z,
            "tile_count": len(tiles),
            "requested_resolution_x_m_per_px": requested_x_m,
            "requested_resolution_y_m_per_px": requested_y_m,
            "source_crs": "EPSG:3857",
            "aligned_crs": "EPSG:27700",
            "alignment": "exact LiDAR raster grid" if reference_raster else "requested BNG bbox",
            "coverage_dependent": True,
            "remote_tile_service": True,
            "attribution": OAM_ATTRIBUTION,
            "description": "OpenAerialMap global mosaic. Real imagery is served at zoom 14+ and may vary in coverage by area.",
        }
        if not tiles:
            errors.append("OpenAerialMap returned no tiles for the requested extent")
            return ImageryResult(None, [], errors, metadata)

        tile_data: dict[tuple[int, int], np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=min(self.workers, len(tiles))) as executor:
            futures = {executor.submit(self._download_tile, z, x, y): (x, y) for x, y in tiles}
            for future in as_completed(futures):
                x, y = futures[future]
                try:
                    tile_xy, cached = future.result()
                    cache_entries.append(cached)
                    from PIL import Image
                    with Image.open(cached.path) as image:
                        tile = np.asarray(image.convert("RGB"), dtype="uint8")
                    if tile.shape[:2] != (TILE_SIZE, TILE_SIZE):
                        from PIL import Image as PILImage
                        tile = np.asarray(PILImage.fromarray(tile).resize((TILE_SIZE, TILE_SIZE), PILImage.Resampling.BILINEAR), dtype="uint8")
                    tile_data[tile_xy] = tile
                except Exception:
                    failed_tiles += 1

        if not tile_data:
            errors.append("OpenAerialMap imagery acquisition failed: no imagery tiles could be downloaded")
            return ImageryResult(None, cache_entries, errors, metadata)

        import rasterio
        from rasterio.transform import from_origin

        world_size = 2.0 * WEB_MERCATOR_HALF
        pixel_size = world_size / ((2**z) * TILE_SIZE)
        mosaic_width = (max_x - min_x + 1) * TILE_SIZE
        mosaic_height = (max_y - min_y + 1) * TILE_SIZE
        mosaic = np.zeros((3, mosaic_height, mosaic_width), dtype="uint8")
        for (tile_x, tile_y), tile in tile_data.items():
            ox = (tile_x - min_x) * TILE_SIZE
            oy = (tile_y - min_y) * TILE_SIZE
            mosaic[:, oy:oy + TILE_SIZE, ox:ox + TILE_SIZE] = np.transpose(tile, (2, 0, 1))

        mosaic_transform = from_origin(
            min_x * 256.0 * pixel_size - WEB_MERCATOR_HALF,
            WEB_MERCATOR_HALF - min_y * 256.0 * pixel_size,
            pixel_size,
            pixel_size,
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            source_path,
            "w",
            driver="GTiff",
            width=mosaic_width,
            height=mosaic_height,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=mosaic_transform,
            compress="deflate",
            photometric="RGB",
        ) as dataset:
            dataset.write(mosaic)

        try:
            if reference_raster is not None and reference_raster.is_file():
                self._align_to_dtm(source_path, reference_raster, aligned_path)
                _render_canvas(aligned_path, output_path, bbox_bng)
            else:
                # Without a reference grid we still need a georeferenced preview.
                with rasterio.open(source_path) as source:
                    source_rgb = source.read([1, 2, 3])
                    from PIL import Image
                    preview = Image.fromarray(np.transpose(source_rgb, (1, 2, 0)))
                    preview.save(output_path, format="PNG", optimize=True)
            metadata.update({
                "enabled": True,
                "cache_hits": sum(1 for item in cache_entries if item.cache_hit),
                "tiles_downloaded": sum(1 for item in cache_entries if not item.cache_hit),
                "tiles_failed": failed_tiles,
                "source_raster": str(source_path),
                "aligned_raster": str(aligned_path) if aligned_path.is_file() else None,
                "preview_path": str(output_path),
                "actual_tile_resolution_m": pixel_size,
            })
            return ImageryResult(output_path, cache_entries, errors, metadata)
        except Exception as exc:
            errors.append(f"OpenAerialMap imagery processing failed: {exc}")
            return ImageryResult(None, cache_entries, errors, metadata)
