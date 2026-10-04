from __future__ import annotations

from pathlib import Path

import numpy as np

from prospector.terrain.anomalies import calculate_archaeological_hillshade, calculate_hillshade


def hillshade(
    dtm_path: Path,
    output_path: Path,
) -> Path:
    """Write the report's high-contrast archaeological hillshade GeoTIFF."""
    import rasterio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dtm_path) as source:
        elevation = source.read(1, masked=True)
        shade = calculate_archaeological_hillshade(
            elevation.filled(np.nan),
            source.res[0],
            source.res[1],
        )
        profile = source.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(shade, 1)
    return output_path
