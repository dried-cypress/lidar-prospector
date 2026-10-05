from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from prospector.providers.http import CachedResponse, HttpClient

# NLS Maps' "LiDAR DTM 50cm-1m (2019-2021)" is a composite background group.
# For England, the NLS viewer delegates to the Environment Agency's 1 m
# composite DTM service. We use the WCS here because it returns elevation
# values, allowing Prospector to calculate its own hillshade consistently.
WCS_URL = (
    "https://environment.data.gov.uk/spatialdata/"
    "lidar-composite-digital-terrain-model-dtm-1m/wcs"
)
DTM_COVERAGE_ID = (
    "13787b9a-26a4-4775-8523-806d13af58fc__"
    "Lidar_Composite_Elevation_DTM_1m"
)


class LidarProvider:
    """Environment Agency DTM provider for the England component of the NLS layer."""

    def __init__(self, client: HttpClient) -> None:
        self.client = client

    @staticmethod
    def validate_capabilities(data: bytes) -> None:
        root = ET.fromstring(data)
        coverage_ids = {
            element.text.strip()
            for element in root.iter()
            if element.tag.endswith("CoverageId")
            and element.text
            and element.text.strip()
        }
        if DTM_COVERAGE_ID not in coverage_ids:
            raise ValueError(
                "Environment Agency WCS capabilities did not contain the expected "
                f"1 m DTM coverage: {DTM_COVERAGE_ID}"
            )

    def capabilities(self) -> tuple[bytes, CachedResponse]:
        cached = self.client.cached_bytes(
            "environment-agency-wcs",
            WCS_URL,
            {"request": "GetCapabilities", "service": "WCS", "version": "2.0.1"},
            suffix=".xml",
            validator=self.validate_capabilities,
        )
        return cached.path.read_bytes(), cached

    @staticmethod
    def coverage_id(capabilities: bytes) -> str:
        root = ET.fromstring(capabilities)
        for element in root.iter():
            if (
                element.tag.endswith("CoverageId")
                and element.text
                and element.text.strip() == DTM_COVERAGE_ID
            ):
                return DTM_COVERAGE_ID
        raise RuntimeError(
            "The expected Environment Agency 1 m DTM coverage was not found in WCS capabilities"
        )

    @staticmethod
    def _validate_geotiff(
        data: bytes,
        expected_bbox: tuple[float, float, float, float],
    ) -> None:
        if not data:
            raise ValueError("Environment Agency returned an empty response")
        if data[:4] not in (b"II*\x00", b"MM\x00*"):
            if data[:1] == b"<":
                preview = data[:500].decode("utf-8", errors="replace")
                raise ValueError(
                    "Environment Agency WCS returned XML instead of GeoTIFF: " + preview
                )
            raise ValueError("Environment Agency WCS response does not appear to be a TIFF")

        try:
            import numpy as np
            import rasterio
        except ImportError as exc:
            raise RuntimeError(
                "LiDAR validation requires Rasterio/NumPy geospatial dependencies. "
                "Install with: pip install -e '.[geo]' or rebuild the Docker trainer image."
            ) from exc

        with rasterio.MemoryFile(data) as memory_file:
            with memory_file.open() as dataset:
                if dataset.driver != "GTiff":
                    raise ValueError(f"LiDAR raster driver is unexpected: {dataset.driver}")
                if dataset.crs is None or dataset.crs.to_epsg() != 27700:
                    raise ValueError(f"LiDAR raster has unexpected CRS: {dataset.crs}")
                if dataset.width <= 0 or dataset.height <= 0 or dataset.count < 1:
                    raise ValueError("LiDAR raster has invalid dimensions or no bands")
                if not np.isclose(dataset.res[0], 1.0, atol=1e-6) or not np.isclose(
                    abs(dataset.res[1]), 1.0, atol=1e-6
                ):
                    raise ValueError(f"LiDAR raster resolution is unexpected: {dataset.res}")

                sample = dataset.read(
                    1,
                    masked=True,
                    out_shape=(min(dataset.height, 100), min(dataset.width, 100)),
                )
                finite = np.isfinite(sample.compressed())
                if not finite.any():
                    raise ValueError("LiDAR raster contains no finite elevation values")

                xmin, ymin, xmax, ymax = expected_bbox
                if (
                    dataset.bounds.right <= xmin
                    or dataset.bounds.left >= xmax
                    or dataset.bounds.top <= ymin
                    or dataset.bounds.bottom >= ymax
                ):
                    raise ValueError("LiDAR raster does not overlap requested area")

    def download_geotiff(
        self,
        bbox: tuple[float, float, float, float],
        destination: Path,
    ) -> tuple[Path, CachedResponse]:
        # Keep the capabilities request in the provenance chain and use the exact
        # current coverage rather than whichever coverage happens to appear first.
        capabilities, _ = self.capabilities()
        coverage = self.coverage_id(capabilities)
        xmin, ymin, xmax, ymax = bbox
        if not xmax > xmin or not ymax > ymin:
            raise ValueError("LiDAR bounding box must have positive width and height")

        params: dict[str, Any] = {
            "service": "WCS",
            "version": "2.0.1",
            "request": "GetCoverage",
            "coverageId": coverage,
            "format": "image/tiff",
            "subset": [f"E({xmin},{xmax})", f"N({ymin},{ymax})"],
        }
        cache = self.client.cached_bytes(
            f"environment-agency-dtm-{coverage}",
            WCS_URL,
            params,
            suffix=".tif",
            validator=lambda data: self._validate_geotiff(data, bbox),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(cache.path.read_bytes())
        return destination, cache
