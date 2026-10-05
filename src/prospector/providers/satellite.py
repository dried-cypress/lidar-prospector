from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer

from prospector.providers.http import CachedResponse, HttpClient

STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
COLLECTION = "sentinel-2-l2a"


@dataclass(frozen=True, slots=True)
class SatelliteResult:
    support_path: Path | None
    preview_path: Path | None
    search_path: Path | None
    scenes: list[dict[str, Any]]
    cache_entries: list[CachedResponse]
    errors: list[str]
    metadata: dict[str, Any]


class SatelliteProvider:
    """Use public Sentinel-2 L2A STAC data as contextual spectral evidence."""

    def __init__(self, client: HttpClient) -> None:
        self.client = client
        self._signed_href_cache: dict[str, str] = {}

    def _search(self, bbox_wgs84: tuple[float, float, float, float]) -> tuple[list[dict[str, Any]], CachedResponse]:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=365 * 3)
        west, south, east, north = bbox_wgs84
        params = {
            "collections": COLLECTION,
            "bbox": f"{west:.8f},{south:.8f},{east:.8f},{north:.8f}",
            "datetime": f"{start.isoformat().replace('+00:00', 'Z')}/{now.isoformat().replace('+00:00', 'Z')}",
            "limit": 60,
            "sortby": "-datetime",
        }
        payload, cached = self.client.cached_json("sentinel-2-stac-search", STAC_SEARCH_URL, params)
        if not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
            raise ValueError("Sentinel-2 STAC response did not contain features")
        return [item for item in payload["features"] if isinstance(item, dict)], cached

    def _select_scenes(self, scenes: list[dict[str, Any]], max_scenes: int = 4) -> list[dict[str, Any]]:
        eligible = []
        for item in scenes:
            props = item.get("properties") or {}
            cloud = props.get("eo:cloud_cover", props.get("s2:cloud_cover"))
            try:
                cloud_value = float(cloud) if cloud is not None else 100.0
            except (TypeError, ValueError):
                cloud_value = 100.0
            if cloud_value > 35.0:
                continue
            eligible.append((cloud_value, item))
        eligible.sort(key=lambda pair: (pair[0], str((pair[1].get("properties") or {}).get("datetime", ""))), reverse=False)
        selected: list[dict[str, Any]] = []
        selected_dates: list[datetime] = []
        for _cloud, item in sorted(eligible, key=lambda pair: str((pair[1].get("properties") or {}).get("datetime", "")), reverse=True):
            value = (item.get("properties") or {}).get("datetime")
            try:
                when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
            if all(abs((when - existing).days) >= 45 for existing in selected_dates):
                selected.append(item)
                selected_dates.append(when)
            if len(selected) >= max_scenes:
                break
        return selected

    def _signed_href(self, href: str) -> str:
        cached = self._signed_href_cache.get(href)
        if cached:
            return cached

        # SAS links are short-lived, so cache them only for the duration of one
        # analysis. This prevents the same asset being signed repeatedly (which
        # is especially important for Sentinel preview generation).
        # Planetary Computer documents that a subscription key raises the rate
        # limit tier; use it when the operator supplies PC_SDK_SUBSCRIPTION_KEY.
        import os

        last_response = None
        headers = {}
        subscription_key = os.getenv("PC_SDK_SUBSCRIPTION_KEY")
        if subscription_key:
            headers["Ocp-Apim-Subscription-Key"] = subscription_key
        for attempt in range(3):
            response = self.client.session.get(
                SIGN_URL,
                params={"href": href},
                headers=headers,
                timeout=self.client.timeout,
            )
            last_response = response
            if response.status_code != 429:
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("href"), str):
                    raise ValueError("Planetary Computer signing API did not return a signed href")
                signed_href = payload["href"]
                self._signed_href_cache[href] = signed_href
                return signed_href

            if attempt == 2:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else 0.5 * (2 ** attempt)
            except (TypeError, ValueError):
                delay = 0.5 * (2 ** attempt)
            time.sleep(min(max(delay, 0.25), 5.0))

        raise RuntimeError(f"Planetary Computer signing failed after retries: {last_response}")

    @staticmethod
    def _asset_href(scene: dict[str, Any], key: str) -> str | None:
        asset = (scene.get("assets") or {}).get(key)
        if isinstance(asset, dict) and isinstance(asset.get("href"), str):
            return asset["href"]
        return None

    def _read_asset_window(
        self,
        href: str,
        dtm_bounds: tuple[float, float, float, float],
        target_shape: tuple[int, int],
    ) -> tuple[np.ndarray, Any, str]:
        import rasterio
        from rasterio.windows import Window, from_bounds

        signed = self._signed_href(href)
        with rasterio.Env(
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.jp2",
            VSI_CACHE=False,
            CPL_VSIL_CURL_CACHE_SIZE=0,
            GDAL_CACHEMAX=64,
        ):
            with rasterio.open(signed) as dataset:
                transformer = Transformer.from_crs("EPSG:27700", dataset.crs, always_xy=True)
                corners = [
                    transformer.transform(dtm_bounds[0], dtm_bounds[1]),
                    transformer.transform(dtm_bounds[2], dtm_bounds[1]),
                    transformer.transform(dtm_bounds[0], dtm_bounds[3]),
                    transformer.transform(dtm_bounds[2], dtm_bounds[3]),
                ]
                xs = [item[0] for item in corners]
                ys = [item[1] for item in corners]
                window = from_bounds(min(xs), min(ys), max(xs), max(ys), dataset.transform)
                window = window.round_offsets().round_lengths()
                window = Window(
                    int(window.col_off),
                    int(window.row_off),
                    max(1, int(window.width)),
                    max(1, int(window.height)),
                )
                masked_data = dataset.read(1, window=window, boundless=True, masked=True)
                data = masked_data.astype("float32").filled(np.nan)
                transform = dataset.window_transform(window)
                return data, transform, str(dataset.crs)

    def _read_asset_window_bands(
        self,
        href: str,
        dtm_bounds: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, Any, str]:
        """Read all RGB bands from one remote COG window using one signed URL."""
        import rasterio
        from rasterio.windows import Window, from_bounds

        signed = self._signed_href(href)
        with rasterio.Env(
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.jp2",
            VSI_CACHE=False,
            CPL_VSIL_CURL_CACHE_SIZE=0,
            GDAL_CACHEMAX=64,
        ):
            with rasterio.open(signed) as dataset:
                if dataset.count < 3:
                    raise ValueError("Satellite visual asset does not contain three colour bands")
                transformer = Transformer.from_crs("EPSG:27700", dataset.crs, always_xy=True)
                corners = [
                    transformer.transform(dtm_bounds[0], dtm_bounds[1]),
                    transformer.transform(dtm_bounds[2], dtm_bounds[1]),
                    transformer.transform(dtm_bounds[0], dtm_bounds[3]),
                    transformer.transform(dtm_bounds[2], dtm_bounds[3]),
                ]
                xs = [item[0] for item in corners]
                ys = [item[1] for item in corners]
                window = from_bounds(min(xs), min(ys), max(xs), max(ys), dataset.transform)
                window = window.round_offsets().round_lengths()
                window = Window(
                    int(window.col_off),
                    int(window.row_off),
                    max(1, int(window.width)),
                    max(1, int(window.height)),
                )
                masked_data = dataset.read(indexes=[1, 2, 3], window=window, boundless=True, masked=True)
                data = masked_data.astype("float32").filled(np.nan)
                return data, dataset.window_transform(window), str(dataset.crs)

    @staticmethod
    def _reproject(
        data: np.ndarray,
        source_transform: Any,
        source_crs: str,
        target_shape: tuple[int, int],
        target_transform: Any,
        target_crs: str = "EPSG:27700",
        *,
        resampling: Any = None,
    ) -> np.ndarray:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        target_shape = (int(target_shape[0]), int(target_shape[1]))
        source = np.asarray(data, dtype="float32")
        source[~np.isfinite(source)] = np.nan
        result = np.full(target_shape, np.nan, dtype="float32")
        reproject(
            source=source,
            destination=result,
            src_transform=source_transform,
            src_crs=source_crs,
            src_nodata=np.nan,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=resampling or Resampling.bilinear,
        )
        return result

    @staticmethod
    def _write_float_raster(path: Path, data: np.ndarray, reference_path: Path) -> Path:
        import rasterio

        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(reference_path) as reference:
            profile = reference.profile.copy()
            profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
            with rasterio.open(path, "w", **profile) as destination:
                destination.write(data.astype("float32"), 1)
        return path

    def _scene_ndvi(
        self,
        scene: dict[str, Any],
        dtm_path: Path,
        dtm_bounds: tuple[float, float, float, float],
        target_shape: tuple[int, int],
        target_transform: Any,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        import rasterio

        red_href = self._asset_href(scene, "B04")
        nir_href = self._asset_href(scene, "B08")
        scl_href = self._asset_href(scene, "SCL")
        if red_href is None or nir_href is None:
            raise ValueError("Sentinel-2 scene lacks B04/B08 assets")
        red, red_transform, red_crs = self._read_asset_window(red_href, dtm_bounds, target_shape)
        nir, nir_transform, nir_crs = self._read_asset_window(nir_href, dtm_bounds, target_shape)
        nir_reprojected = self._reproject(nir, nir_transform, nir_crs, red.shape, red_transform)
        ndvi = np.divide(
            nir_reprojected - red,
            nir_reprojected + red,
            out=np.full_like(red, np.nan, dtype="float32"),
            where=np.abs(nir_reprojected + red) > 1e-6,
        )
        # Sentinel-2 surface reflectance assets are commonly stored with a
        # scale factor, but NDVI is invariant to a common multiplicative scale.
        if scl_href:
            scl, scl_transform, scl_crs = self._read_asset_window(scl_href, dtm_bounds, target_shape)
            scl_resampled = self._reproject(
                scl,
                scl_transform,
                scl_crs,
                red.shape,
                red_transform,
                resampling=rasterio.enums.Resampling.nearest,
            )
            # Cloud shadow, medium/high probability cloud, cirrus and snow.
            # Avoid casting NaN nodata values directly to int16, which emits
            # RuntimeWarning on current NumPy versions.
            scl_classes = np.zeros(scl_resampled.shape, dtype="int16")
            scl_valid = np.isfinite(scl_resampled)
            scl_classes[scl_valid] = np.rint(scl_resampled[scl_valid]).astype("int16")
            ndvi[np.isin(scl_classes, [3, 8, 9, 10, 11]) & scl_valid] = np.nan
        target_ndvi = self._reproject(ndvi, red_transform, red_crs, target_shape, target_transform)
        return target_ndvi, ndvi

    @staticmethod
    def _spectral_anomaly(ndvi: np.ndarray, pixels_per_metre: float = 1.0) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        valid = np.isfinite(ndvi)
        if valid.sum() < 100:
            return np.zeros_like(ndvi, dtype="float32")
        filled = ndvi.copy()
        median = float(np.nanmedian(filled))
        filled[~valid] = median
        sigma = max(3.0, 25.0 * pixels_per_metre)
        background = gaussian_filter(filled, sigma=sigma, mode="nearest")
        residual = np.abs(filled - background)
        residual[~valid] = np.nan
        centre = float(np.nanmedian(residual))
        mad = float(np.nanmedian(np.abs(residual - centre)))
        scale = max(1e-5, 1.4826 * mad)
        return np.clip(residual / (6.0 * scale), 0.0, 1.0).astype("float32")

    def _preview_rgb(
        self,
        scene: dict[str, Any],
        dtm_bounds: tuple[float, float, float, float],
        target_shape: tuple[int, int],
        target_transform: Any,
        output_path: Path,
    ) -> Path | None:
        """Create an AOI-aligned RGB PNG, preferring Sentinel's pre-rendered visual asset.

        Sentinel-2 L2A exposes a `visual` asset specifically intended for true-colour
        rendering. Using it means one signing request instead of independently
        signing B04/B03/B02 during preview generation.
        """
        import rasterio
        from rasterio.io import MemoryFile

        visual_href = self._asset_href(scene, "visual")
        if visual_href is not None:
            data, source_transform, source_crs = self._read_asset_window_bands(
                visual_href, dtm_bounds
            )
            bands: list[np.ndarray] = []
            for band in data:
                bands.append(
                    self._reproject(
                        band,
                        source_transform,
                        source_crs,
                        target_shape,
                        target_transform,
                    )
                )
            stack = np.stack(bands, axis=-1)
        else:
            # Compatibility fallback for STAC items without the visual asset.
            arrays: list[np.ndarray] = []
            for key in ("B04", "B03", "B02"):
                href = self._asset_href(scene, key)
                if href is None:
                    return None
                array, source_transform, source_crs = self._read_asset_window(
                    href, dtm_bounds, target_shape
                )
                arrays.append(
                    self._reproject(
                        array, source_transform, source_crs, target_shape, target_transform
                    )
                )
            stack = np.stack(arrays, axis=-1)

        valid = np.isfinite(stack).all(axis=2)
        if not valid.any():
            return None
        image = np.nan_to_num(stack, nan=np.nanmedian(stack, axis=(0, 1)))
        low = np.nanpercentile(image, 2, axis=(0, 1))
        high = np.nanpercentile(image, 98, axis=(0, 1))
        span = np.maximum(high - low, 1e-6)
        image = np.clip((image - low) / span, 0.0, 1.0)
        image_u8 = np.rint(image * 255.0).astype("uint8")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = image_u8.shape[:2]
        with MemoryFile() as memory:
            with memory.open(driver="PNG", height=height, width=width, count=3, dtype="uint8") as png:
                png.write(np.moveaxis(image_u8, 2, 0))
            output_path.write_bytes(memory.read())
        return output_path

    def acquire(self, bbox_bng: tuple[float, float, float, float], dtm_path: Path, run_dir: Path) -> SatelliteResult:
        import rasterio

        errors: list[str] = []
        cache_entries: list[CachedResponse] = []
        search_path: Path | None = None
        preview_path: Path | None = None
        support_path: Path | None = None
        scenes: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {"provider": "Microsoft Planetary Computer / Copernicus Sentinel-2 L2A"}
        run_dir.mkdir(parents=True, exist_ok=True)
        source_dir = run_dir / "sources" / "satellite"
        source_dir.mkdir(parents=True, exist_ok=True)

        try:
            to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
            xmin, ymin, xmax, ymax = bbox_bng
            points = [
                to_wgs84.transform(x, y)
                for x, y in ((xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax))
            ]
            bbox_wgs84 = (
                min(point[0] for point in points),
                min(point[1] for point in points),
                max(point[0] for point in points),
                max(point[1] for point in points),
            )
            all_scenes, search_cache = self._search(bbox_wgs84)
            cache_entries.append(search_cache)
            scenes = self._select_scenes(all_scenes)
            search_path = source_dir / "sentinel-2-stac-search.json"
            search_path.write_text(
                json.dumps(
                    {"collection": COLLECTION, "bbox_wgs84": bbox_wgs84, "scenes": all_scenes},
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            if not scenes:
                raise RuntimeError("No sufficiently clear Sentinel-2 scenes were found for the study area")

            with rasterio.open(dtm_path) as dtm:
                target_shape = (int(dtm.height), int(dtm.width))
                target_transform = dtm.transform
                target_res = float(max(abs(dtm.res[0]), abs(dtm.res[1])))

            support_arrays: list[np.ndarray] = []
            successful_scenes: list[dict[str, Any]] = []
            for scene in scenes:
                scene_id = str(scene.get("id") or "unknown-scene")
                try:
                    ndvi, _native_ndvi = self._scene_ndvi(
                        scene,
                        dtm_path,
                        bbox_bng,
                        target_shape,
                        target_transform,
                    )
                    support_arrays.append(
                        self._spectral_anomaly(
                            ndvi,
                            pixels_per_metre=max(1.0, 1.0 / target_res),
                        )
                    )
                    successful_scenes.append(scene)
                except Exception as exc:
                    errors.append(f"Sentinel-2 scene {scene_id} processing failed: {exc}")

            if not support_arrays:
                raise RuntimeError("No selected Sentinel-2 scene could be processed successfully")

            support = np.nanmedian(np.stack(support_arrays, axis=0), axis=0).astype("float32")
            support_path = self._write_float_raster(
                run_dir / "terrain" / "satellite-support.tif",
                support,
                dtm_path,
            )

            metadata.update(
                {
                    "scene_count": len(scenes),
                    "successful_scene_count": len(successful_scenes),
                    "failed_scene_count": len(scenes) - len(successful_scenes),
                    "selected_scenes": [
                        {
                            "id": item.get("id"),
                            "datetime": (item.get("properties") or {}).get("datetime"),
                            "cloud_cover": (item.get("properties") or {}).get("eo:cloud_cover"),
                        }
                        for item in scenes
                    ],
                    "support_description": "Median multi-scene local NDVI anomaly, used as contextual evidence rather than an archaeological classifier.",
                    "preview_description": "Sentinel-2 visual preview intentionally disabled in v0.4.4; high-resolution Esri World Imagery is the visual base layer.",
                    "preview_asset": None,
                }
            )
        except Exception as exc:
            errors.append(f"Sentinel-2 acquisition/processing failed: {exc}")

        return SatelliteResult(
            support_path,
            preview_path,
            search_path,
            scenes,
            cache_entries,
            errors,
            metadata,
        )

