## 0.4.0

- Replace the threshold-only terrain detector with a hybrid multi-signal discovery engine combining multi-scale relief persistence, Hough-supported linear structure, annular/ring response, ridge/valley morphology, local texture/coherence, Historic England morphology similarity, and per-AOI unsupervised IsolationForest terrain novelty.
- Replace low/medium/high detector sensitivity with a 1–10 scale; level 5 is the default balanced research setting and level 10 is intentionally exploratory/high-recall.
- Add transparent detector diagnostic rasters for discovery score, terrain novelty, ring response, and ridge/valley response.
- Add optional Sentinel-2 imagery as a switchable report base layer and fix integer-raster reprojection failures in satellite processing.
- Add optional coordinate reverse-geocoding so reports show a human-readable location name while retaining coordinates as the authoritative study input.
- Move interactive layer controls above the image into a collapsible panel and expand report themes with Rose, Lavender, Mint, and Sky pastel themes alongside Light and Dark.
- Add scikit-learn as a runtime dependency for the new unsupervised anomaly-ranking stage.

## 0.3.7

- Stop candidate generation before Historic England detailed mapping and monument extents are labelled, preventing detector geometry from tracing known archaeological outlines or producing rings/halos around them.
- Treat Monument Extents as a hard discovery exclusion while retaining detections outside the registered envelope; detailed mapping uses a smaller exclusion buffer so a newly detected feature can cross an existing mapped line without becoming that line.
- Make candidate generation independent of Historic England availability; zero mapped records remain a valid discovery input.
- Promote long linear earthworks to a first-class candidate generator using multi-scale linear response and Hough-seeded coherent lines.
- Increase linear evidence weighting so long, straight banks, ditches and scarps compete with compact anomalies.
- Add `--sensitivity low|medium|high` and `--workers` controls with conservative automatic threading.
- Add clickable, labelled anomaly markers in the interactive HTML report and a report theme toggle with a modern dark mode.
- Keep switchable LiDAR hillshade, anomaly relief and Sentinel-2 report bases with Historic England and anomaly overlays.
- Harden Sentinel-2 raster window handling by coercing all requested windows and shapes to integer pixel dimensions.

## 0.3.6

- Rasterise Historic England project areas, monument extents and detailed mapping more prominently and consistently in `lidar-aim.png` and `lidar-anomalies-aim.png`, so the PNG itself remains a complete archaeological reference rather than relying on the HTML SVG overlay.
- Added switchable local HTML evidence maps for LiDAR hillshade, LiDAR anomaly relief and Sentinel-2 preview imagery with Historic England and anomaly vectors above them.
- Added a Hough-seeded straight-line detector to recover long, coherent field banks, ditches and scarps that fragment in the pixel anomaly mask.
- Stop treating Monument Extents as an exclusion mask; they are broad envelopes and may contain unmapped individual features that should remain discoverable.
- Reduced the default known-feature exclusion buffer and increased the maximum candidate area to avoid suppressing broad linear earthworks.
- Satellite Sentinel-2 COG reads now disable GDAL's persistent remote cache; only the derived support raster, preview and STAC metadata are retained.

## 0.3.5

- Fix incomplete Historic England Aerial Investigation and Mapping acquisition by paging all FeatureServer query responses up to the published 2,000-record page limit.
- Resolve Historic England layers by their published names, with the historical numeric IDs retained only as a compatibility fallback.
- Record per-layer page/record counts and explicit truncation status in run metadata and command output.
- Ensure the complete retrieved `Detailed_Mapping` feature set is the input to the existing anomaly exclusion and Historic England morphology-similarity stages.
- Preserve the v0.3.4 terrain/anomaly ranking algorithm unchanged so detector-quality improvements can be evaluated separately.
- Note the Historic England service limitation that earlier hand-drawn projects remain raster-only and are not included in the downloadable vector FeatureServer dataset.

## 0.3.4

- Fix the v0.3.3 anomaly-output crash caused by undefined Historic England context variables in the LiDAR-only anomaly renderer.
- Keep all three Historic England Aerial Archaeology Mapping layers in both the AIM image and combined prospects + AIM image: project areas, monument extents and detailed mapping.
- Preserve the v0.3.3 HTML/SVG equal-aspect overlay alignment unchanged.
- Eliminate the Sentinel-2 SCL invalid-cast runtime warning when cloud-mask rasters contain nodata.
- Retry transient Planetary Computer asset-signing HTTP 429 responses before failing optional satellite preview processing.

## 0.3.3

- Apply all three Historic England Aerial Archaeology Mapping layers to LiDAR AIM outputs: project areas, monument extents and detailed mapping.
- Retain detailed mapping as the archaeological reference layer used by anomaly scoring.
- Fix horizontal distortion of interactive HTML overlays by matching the SVG viewport to the equal-aspect map area inside the PNG.
- Fix Sentinel-2 processing warning caused by a missing local `rasterio` import.
- Expose the three Historic England layers as separate GeoJSON run artefacts.
- Print acquisition warnings with their actual messages at the end of an analysis.

# Changelog

## Unreleased

## 0.4.0 — Hybrid terrain-pattern detector

- Replace the low/medium/high detector control with a 1–10 sensitivity scale; default `5` balances selectivity and recall, while `10` is deliberately exploratory.
- Add multi-scale annular/ring response for closed banks, enclosures and mound-like forms.
- Add multi-scale ridge/valley response and local texture/coherence channels for scarps, hollow ways, terraces and subtle earthworks.
- Add unsupervised per-AOI `IsolationForest` terrain novelty scoring using the combined LiDAR feature stack; no AI API calls or remote model inference are used.
- Keep Historic England as contextual evidence/exclusion rather than making known archaeology a prerequisite for discovery.
- Expose the new detector channels, novelty metadata and reasons in candidate GeoJSON and the HTML report.
- Add optional reverse-geocoded place naming to reports with cached provenance and `PROSPECTOR_GEOCODER_URL` override.
- Move interactive map controls above the image so they cannot obscure anomaly geometry; make the layer menu collapsible.
- Expand report themes to light, dark, rose, lavender, mint and sky pastel palettes.
- Fix Sentinel-2 processing of integer source rasters by normalising source data to floating point before NaN-aware reprojection.
- Make satellite processing resilient to a single bad scene and retain a usable support raster/preview when other selected scenes succeed.
- Add `scikit-learn` as a runtime dependency for the unsupervised detector.

## 0.3.7

- Stop candidate generation before Historic England detailed mapping and monument extents are labelled, preventing detector geometry from tracing known archaeological outlines or producing rings/halos around them.
- Treat Monument Extents as a hard discovery exclusion while retaining detections outside the registered envelope; detailed mapping uses a smaller exclusion buffer so a newly detected feature can cross an existing mapped line without becoming that line.
- Make candidate generation independent of Historic England availability; zero mapped records remain a valid discovery input.
- Promote long linear earthworks to a first-class candidate generator using multi-scale linear response and Hough-seeded coherent lines.
- Increase linear evidence weighting so long, straight banks, ditches and scarps compete with compact anomalies.
- Add `--sensitivity low|medium|high` and `--workers` controls with conservative automatic threading.
- Add clickable, labelled anomaly markers in the interactive HTML report and a report theme toggle with a modern dark mode.
- Keep switchable LiDAR hillshade, anomaly relief and Sentinel-2 report bases with Historic England and anomaly overlays.
- Harden Sentinel-2 raster window handling by coercing all requested windows and shapes to integer pixel dimensions.
- Prefer the OS Data Hub documented `OS_DATAHUB_API_KEY` environment variable for OS Features API authentication; retain `OS_API_KEY` as a backwards-compatible fallback.
- Clarify that the OS Features API key-header authentication used by Prospector does not require the API project secret.


## 0.3.1 — Fast AOI-scoped OS context

- Prefer the AOI-scoped Ordnance Survey Features API / OS Open Zoomstack when `OS_API_KEY` is configured.
- Do not silently trigger a large OS OpenMap Local 100 km-grid download during a normal small-area analysis.
- Keep `--os-download` as an explicit slow-download path and `--os-data` for local/offline data.
- Report OS context elapsed time.
- Preserve the V0.2.1 LiDAR rendering pipeline unchanged.


## 0.3.0 — Context-aware anomaly ranking

- Keep the V0.2.1 LiDAR visualisation settings unchanged.
- Retain the NLS/Environment Agency LiDAR source and Historic England AIM acquisition pipeline.
- Add multi-scale persistence scoring so a feature must demonstrate coherent response across several LiDAR scales to receive a strong persistence score.
- Add candidate morphology scoring for elongation, circularity, rectangularity, convexity and closed/ring geometry.
- Add modern-feature context using Ordnance Survey OS OpenMap Local.
- Add `--os-data` for local OS OpenMap Local archives/directories.
- Add OpenStreetMap as a supplementary provider for paths, tracks, fences, hedges and buildings.
- Add a soft modernity penalty with much stronger suppression for buildings/roads and softer treatment of paths/boundaries.
- Add Sentinel-2 L2A discovery through the Microsoft Planetary Computer STAC API.
- Add multi-scene NDVI anomaly support and a latest-scene RGB preview.
- Add deterministic geometry similarity against known Historic England AIM features.
- Expand candidate GeoJSON properties and HTML reporting with component scores and reasons.
- Change the project licence from MIT to GNU GPL v3.
- Add provider/context regression tests.
- Do not change the known HTML/SVG overlay alignment implementation in this release.
