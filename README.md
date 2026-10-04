# Prospector

Prospector is a reproducible archaeological landscape prospection tool for combining LiDAR terrain data with heritage, modern-feature and satellite context.

## Version 0.3.7

V0.3.7 keeps the proven LiDAR visualisation settings while making the raster archaeological overlays complete and substantially improving the anomaly detector without replacing the current architecture.

The detector now:

- retains high-recall multi-scale LiDAR anomaly generation
- measures cross-scale persistence instead of rewarding a single extreme scale
- scores candidate morphology, including compactness, elongation, circularity, rectangularity and ring/closed geometry
- acquires Ordnance Survey OS OpenMap Local context where available
- optionally accepts local OS OpenMap Local data with `--os-data`
- supplements OS transport/building coverage with OpenStreetMap paths, tracks, fences, hedges and buildings
- creates a soft modernity penalty rather than blindly deleting every mapped path or boundary
- acquires multi-scene Sentinel-2 L2A imagery from the Microsoft Planetary Computer public STAC catalogue
- derives a multi-scene satellite spectral-support raster from local NDVI anomalies
- compares candidate geometry against known Historic England AIM feature geometry as a deterministic similarity prior
- exposes the component scores and reasons in GeoJSON and the HTML report

This is still a deterministic contextual ranker, **not a trained ML model**. The output remains archaeological research leads rather than confirmed sites.

## Locked LiDAR visualisation

The established archaeological hillshade presentation is kept stable in V0.3.7:

- 8-direction multidirectional hillshade at 38° altitude
- blended 70% multidirectional / 30% conventional north-west illumination at 42° altitude
- robust 2nd–98th percentile contrast stretch
- power 0.90 mid-tone adjustment
- restrained 0.14 elevation tint in the LiDAR/AIM image
- the anomaly/combined image renderer continues to use the same hillshade and relief visualisation settings as V0.2.1

The generated PNG overlays are therefore a regression target for future versions. V0.3.7 makes the underlying PNG renderer itself authoritative: Historic England layers are rasterised directly into the generated PNGs, while the HTML report additionally exposes the same vectors as toggleable overlays.

## LiDAR source

The NLS Maps viewer presents an England/Wales/Scotland **LiDAR DTM 50cm-1m** composite. For England, Prospector obtains numeric DTM elevation from the Environment Agency WCS service so the application can render its own archaeology-focused hillshade rather than treating a rendered RGB tile as elevation.

The pinned England coverage is:

```text
13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m
```

## Historic England

Historic England's **Detailed_Mapping** service remains the authoritative known-archaeology layer, while Monument_Extents and Project_Area are also retrieved for faithful Aerial Archaeology Mapping visualisation. Prospector consumes the underlying ArcGIS FeatureServer directly rather than scraping the Experience Builder application, so the data acquisition is independent of the Explorer's zoom-dependent presentation. V0.3.5 resolved the live FeatureServer by published layer name and paginates all vector results, preventing a full first response from hiding additional features. Historic England also notes that earlier hand-drawn projects remain raster-only and are not currently downloadable from this vector service.

1. Detailed_Mapping features are excluded from candidate generation using a small buffer so the detector does not reproduce existing outlines;
2. Monument_Extents form a hard registered-area exclusion, while candidates crossing a mapped detailed feature can retain their unregistered outside geometry;
3. known feature geometry provides only a weak contextual similarity signal and cannot manufacture candidates;
4. a dedicated multi-scale linear response and probabilistic Hough seeding recover long straight banks, ditches and scarps;
5. candidate generation remains independent of Historic England and works with zero known records.

Historic England's Aerial Investigation Mapping data depict archaeology identified, mapped and recorded from aerial photographs and other aerial sources across England. Historic England's current 2026 standards call for a systematic and integrated GIS approach combining aerial interpretation and lidar.

## Modern context

V0.3.x treats modern mapping as contextual evidence rather than a universal mask.

The strongest suppression is applied to:

- buildings
- roads
- major tracks

Softer suppression is applied to:

- footpaths
- minor paths
- fences
- hedges
- other mapped boundaries

This distinction matters because archaeological routeways and boundaries can survive beneath or alongside modern ones.

### Ordnance Survey

OS Features API / OS Open Zoomstack is the preferred automatic OS source when `OS_API_KEY` is configured because it returns only AOI-intersecting features. The slower OS OpenMap Local 100 km National Grid download remains available with `--os-download`, while `--os-data` supports local/offline OML data.

When automatic OS access is unsuitable, supply local OS data:

```bash
prospector analyse --latitude 50.8657 --longitude -0.2405 \
  --os-data /path/to/os-openmap-local-tq.zip
```

A GeoPackage, Shapefile ZIP or extracted vector directory can be supplied. V0.3.7 also queries OpenStreetMap as a supplementary source for paths, tracks, fences, hedges and buildings.

## Satellite context

Prospector uses Sentinel-2 Level-2A data via the Microsoft Planetary Computer STAC API. The service exposes a public STAC catalogue and Sentinel-2 data without requiring a user account for catalogue discovery; file access uses short-lived signed asset URLs.

For each run, Prospector selects several relatively clear observations across time rather than trusting a single image. It calculates local NDVI anomaly support for each scene and uses the median support across selected scenes when ranking candidates.

The satellite signal is deliberately **supporting evidence**, not a classifier. Cropmarks and vegetation responses may strengthen a candidate, but modern vegetation boundaries can also produce a response and are handled separately through mapped context. Satellite scenes are streamed as remote COG windows; Prospector stores the derived support raster, a report preview and STAC provenance rather than copying complete Sentinel-2 scenes into every run.

## Detection model

The V0.3.7 detector is intentionally transparent and reproducible:

```text
LiDAR DTM
   |
   +--> multi-scale local relief
   +--> robust z-score anomaly field
   +--> cross-scale persistence
   +--> multi-scale linear/edge response + straight-line Hough seeding
   +--> mask registered archaeology before components are built
   +--> connected candidate regions
            |
            +--> geometry / morphology
            +--> modern context penalty
            +--> satellite spectral support
            +--> Historic England geometry similarity
                        |
                        v
                  final ranked score
```

The final score is a weighted contextual rank, not a probability. Its components are recorded in the candidate GeoJSON and report so that false positives can be diagnosed rather than silently discarded.

## Run outputs

Each run is under `project/runs/<run-id>/`:

```text
sources/
  historic-england-detailed-mapping.geojson
  historic-england-monument-extents.geojson
  historic-england-project-areas.geojson
  modern-context.geojson
  os-openmap/
  satellite/

terrain/
  dtm.tif
  hillshade.tif
  modern-context-score.tif
  satellite-support.tif

overlays/
  lidar-aim.png
  lidar-anomalies.png
  lidar-anomalies-aim.png
  satellite-latest.png
  lidar-base.png
  lidar-anomalies-base.png

candidates/
  terrain-anomalies.geojson

reports/
  report.html

run.json
```

The exact set of files depends on whether external providers succeeded.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

## Analyse

```bash
prospector analyse --latitude 50.8657 --longitude -0.2405
```

Detection sensitivity can be changed per run:

```bash
prospector analyse --latitude 50.8657 --longitude -0.2405 --sensitivity low
prospector analyse --latitude 50.8657 --longitude -0.2405 --sensitivity high
```

`low` favours larger, stronger structures; `medium` is the normal research setting; `high` broadens the search to subtle terrain signatures. `--workers 0` uses a conservative automatic detector thread count; a positive value overrides it.

Useful offline/reduced-data options:

```bash
prospector analyse --latitude 50.8657 --longitude -0.2405 --no-context
prospector analyse --latitude 50.8657 --longitude -0.2405 --no-osm
prospector analyse --latitude 50.8657 --longitude -0.2405 --no-satellite
```

## Licensing

Prospector source code is released under **GNU GPL v3**.

The external datasets remain under their respective upstream licences/terms. The application records their provenance rather than relicensing the underlying datasets as GPL.

## Development philosophy

V0.3.7 is deliberately the improved V1 detector. A future V2 can use this deterministic system as the labelled-data and hard-negative generation layer for a learned multimodal model. Human-reviewed candidate outcomes should eventually become the project's most valuable training data.

### Fast OS context

Set `OS_DATAHUB_API_KEY` to use AOI-scoped OS Features API queries for buildings, roads, rail and water. The OS Features API can authenticate with the API key directly; the API project secret is not required for this request method. `OS_API_KEY` is retained as a legacy fallback. Without a key, V0.3.4 deliberately skips the large OpenMap Local tile download rather than blocking a normal analysis. Use `--os-download` when you explicitly want the legacy OpenMap Local download path. The OS API key is read from `OS_DATAHUB_API_KEY`.
