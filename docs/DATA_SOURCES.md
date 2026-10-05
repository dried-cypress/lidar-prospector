# Data sources

## Historic England Aerial Investigation & Mapping

`https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/arcgis/rest/services/HE_AIM_data/FeatureServer`

Prospector queries the published `Detailed_Mapping`, `Monument_Extents` and `Project_Area` layer names in EPSG:27700. Historic numeric layer IDs are retained only as a compatibility fallback. Each layer query is explicitly paginated at the service's 2,000-record limit so a full first response cannot silently truncate an AOI. The layer contains archaeological features identified from aerial imagery and airborne LiDAR. The API supplies descriptive metadata including `PERIOD`, `MONUMENT_TYPE`, `EVIDENCE_1`, `SOURCE_1`, `HE_UID`, `HER_NO`, and direct record links.

Detailed AIM features remain the authoritative exclusion layer for already-mapped archaeology and are also used as a local morphology-similarity reference during V0.3.5 ranking.

## National Library of Scotland / Environment Agency LiDAR

The NLS Maps viewer labels the combined terrain background **LiDAR DTM 50cm-1m (2019-2021)**. Prospector uses the Environment Agency's current WCS service for the England component so that it receives numeric elevation values.

WCS:

`https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs`

Pinned 1 m coverage:

`13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m`

All LiDAR and Historic England requests use British National Grid (EPSG:27700). All contextual rasters are snapped to the exact LiDAR raster grid after acquisition.

## Ordnance Survey OS OpenMap Local

OS OpenMap Local is an open vector/raster contextual mapping product for Great Britain, updated twice per year and intended for analytical/contextual use.

Prospector discovers the current OpenMap Local product through the OS Downloads API and requests the 100 km National Grid tile(s) covering the study area. Context is classified into buildings, roads, tracks, paths and boundaries where the supplied OS feature/layer metadata permits.

A local archive or extracted data can be passed using `--os-data` when automatic access is unsuitable.

## OpenStreetMap

OpenStreetMap is used only as supplementary modern context. Prospector queries the Overpass API for highways, buildings, barriers and hedges and classifies these as road, track, path, building or boundary features.

OSM data should be treated under its upstream ODbL terms and is not redistributed by Prospector as a new licensed dataset.

## Sentinel-2

Prospector uses the Microsoft Planetary Computer public STAC catalogue for Sentinel-2 Level-2A imagery:

`https://planetarycomputer.microsoft.com/api/stac/v1/search`

Selected assets use short-lived signed URLs. Sentinel-2 is optional spectral evidence only; it is not used as the interactive report base image. The high-resolution visual base is provided separately by Esri World Imagery. Satellite data are contextual evidence and are never presented as proof of archaeology.

## LiDAR terrain anomaly detector

V0.3.5 begins with the V0.2.1 multi-scale Local Relief Model detector but adds:

- 8, 16, 32 and 64 m relief scales
- robust median/MAD z-scores
- cross-scale persistence
- morphology metrics
- modern-feature contextual penalty
- multi-scene satellite spectral support
- LiDAR terrain-signature similarity to known Historic England features

The result is a deterministic contextual ranker. Candidates remain research leads, not archaeological identifications.


## High-resolution visual imagery

Prospector downloads an AOI-aligned visual image from Esri World Imagery using the ArcGIS World Imagery map service export endpoint. World Imagery is a compilation of satellite and aerial imagery; available resolution varies by source and location, with high-resolution sources used in many areas.

The visual image is presentation/context data only. It is not used as the primary archaeology detector input; the detector works from the numeric LiDAR DTM. The report records Esri attribution alongside the imagery.


## V0.5 local training corpus

V0.5 can ingest the three current public HE AIM layers in bounded EPSG:27700 chunks: `Detailed_Mapping`, `Monument_Extents` and `Project_Area`. All three are stored in PostGIS. `Detailed_Mapping` supplies positive archaeological examples; the other layers remain contextual and are used when constructing background masks.

For each positive feature, Prospector extracts the corresponding numeric LiDAR terrain directly from the DTM and creates examples at several physical scales with rotation augmentation. The model therefore learns terrain morphology rather than RGB appearance, and training/validation groups keep variants of the same source feature together.
