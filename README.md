# Prospector

Prospector is a reproducible archaeological landscape prospection tool for combining LiDAR terrain data with heritage, modern-feature and satellite context.

## Version 0.4.4

V0.4.4 is the detector-focused release, with Python 3.10 compatibility restored. It keeps the established LiDAR visualisation while adding a hybrid terrain-pattern detector designed to find archaeological-looking structures that do not resemble a single bright local-relief blob.

The detector now:

- retains high-recall multi-scale LiDAR anomaly generation
- measures cross-scale persistence instead of rewarding a single extreme scale
- scores candidate morphology, including compactness, elongation, circularity, rectangularity and ring/closed geometry
- acquires Ordnance Survey OS OpenMap Local context where available
- optionally accepts local OS OpenMap Local data with `--os-data`
- supplements OS transport/building coverage with OpenStreetMap paths, tracks, fences, hedges and buildings
- creates a soft modernity penalty rather than blindly deleting every mapped path or boundary
- uses Sentinel-2 only as optional spectral context and downloads high-resolution Esri World Imagery for visual inspection
- derives a multi-scene satellite spectral-support raster from local NDVI anomalies
- compares candidate geometry against known Historic England AIM feature geometry as a deterministic LiDAR terrain-signature reference prior
- measures annular/ring response for enclosures, barrows and bank-like closed forms
- measures ridge/valley response for scarps, hollow ways, terraces and subtle earthworks
- measures local texture and surface coherence so visually distinctive terrain is not lost by a single threshold
- uses an unsupervised `IsolationForest` over the terrain feature stack to surface unusual combinations learned from the current AOI
- exposes every detector channel, threshold and reason in GeoJSON and the HTML report

The ML component is deliberately **not an AI service**. V0.4.4 combines an unsupervised `IsolationForest` terrain-novelty model with a supervised `RandomForestClassifier` trained locally from Historic England Detailed_Mapping positives versus non-HE background within the current AOI. It is a transparent terrain-likelihood aid rather than a claim of definitive archaeological classification; a reviewed cross-site corpus remains the next step for robust generalisation.

## Locked LiDAR visualisation

The established archaeological hillshade presentation is kept stable in V0.4.4:

- 8-direction multidirectional hillshade at 38° altitude
- blended 70% multidirectional / 30% conventional north-west illumination at 42° altitude
- robust 2nd–98th percentile contrast stretch
- power 0.90 mid-tone adjustment
- restrained 0.14 elevation tint in the LiDAR/AIM image
- the anomaly/combined image renderer continues to use the same hillshade and relief visualisation settings as V0.2.1

The generated PNG overlays are therefore a regression target for future versions. V0.4.4 keeps the underlying PNG renderer authoritative: Historic England layers are rasterised directly into the generated PNGs, while the HTML report additionally exposes the same vectors as toggleable overlays.

## LiDAR source

The NLS Maps viewer presents an England/Wales/Scotland **LiDAR DTM 50cm-1m** composite. For England, Prospector obtains numeric DTM elevation from the Environment Agency WCS service so the application can render its own archaeology-focused hillshade rather than treating a rendered RGB tile as elevation.

The pinned England coverage is:

```text
13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m
```

## Historic England

Historic England's **Detailed_Mapping** service remains the authoritative known-archaeology layer, while Monument_Extents and Project_Area are also retrieved for faithful Aerial Archaeology Mapping visualisation. Prospector consumes the underlying ArcGIS FeatureServer directly rather than scraping the Experience Builder application, so the data acquisition is independent of the Explorer's zoom-dependent presentation. V0.3.5 resolved the live FeatureServer by published layer name and paginates all vector results, preventing a full first response from hiding additional features. Historic England also notes that earlier hand-drawn projects remain raster-only and are not currently downloadable from this vector service.

1. Detailed_Mapping features are used as positive detector references first, then their mapped areas are removed from the final discovery candidate set using a small buffer;
2. Monument_Extents form a hard registered-area exclusion only after detector validation, while candidates crossing a mapped detailed feature can retain their unregistered outside geometry;
3. the detector compares LiDAR-derived terrain signatures against known HE references; the HE reference bank does not replace candidate generation or manufacture terrain evidence;
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


### High-resolution visual imagery

Prospector uses **[Esri World Imagery](https://developers.arcgis.com/rest/basemap-styles/service-data/)** as the visual aerial/satellite base layer in the interactive report. The application requests an AOI-aligned PNG from the World Imagery map service at up to 2048 pixels across, so a 1 km study area is rendered at roughly 0.49 m per output pixel. Source resolution varies by location: World Imagery is a compilation of satellite and aerial imagery, with high-resolution sources used in many areas.

Sentinel-2 is retained only as optional spectral context for detector scoring. It is no longer used as the visual base layer because its 10 m imagery is not appropriate for detailed inspection of small archaeological earthworks.

### OS Data Hub API signup

Prospector can use the **Ordnance Survey (OS) Features API** for fast, AOI-scoped modern-feature context. The implementation uses the OS Features API WFS endpoint with OS Open Zoomstack layers for local buildings, roads, rail and water. This is separate from the optional OS OpenMap Local download path.

Create an account and API project through the [OS Data Hub](https://osdatahub.os.uk/). OS requires users to register for the applicable API plan and account before using the API services. Each API project has its own API key.

For Prospector, export the API key as:

```bash
export OS_DATAHUB_API_KEY="YOUR_OS_DATA_HUB_API_KEY"
```

Prospector reads `OS_DATAHUB_API_KEY` for the OS Features API. `OS_API_KEY` is also accepted as a backwards-compatible legacy variable. The key is sent to the OS Features API as the `key` HTTP header; an OS API project secret is not required by this request path.

You can check the current OS API service status on the OS Data Hub before troubleshooting a failed context acquisition.

The API key is optional: without one, the normal analysis still runs using LiDAR and the other available context sources. The slower OS OpenMap Local download can be explicitly enabled with `--os-download`.

### Ordnance Survey

OS Features API / OS Open Zoomstack is the preferred automatic OS source when `OS_DATAHUB_API_KEY` is configured because it returns only AOI-intersecting features. The slower OS OpenMap Local 100 km National Grid download remains available with `--os-download`, while `--os-data` supports local/offline OML data.

When automatic OS access is unsuitable, supply local OS data:

```bash
prospector analyse --latitude 50.8657 --longitude -0.2405 \
  --os-data /path/to/os-openmap-local-tq.zip
```

A GeoPackage, Shapefile ZIP or extracted vector directory can be supplied. V0.4.4 also queries OpenStreetMap as a supplementary source for paths, tracks, fences, hedges and buildings.

## Coordinate naming

The CLI remains coordinate-driven. V0.4.4 also performs an optional reverse-geocode lookup so the report can display a human-readable location name while retaining the exact latitude/longitude as the authoritative input. The default provider is Nominatim; set `PROSPECTOR_GEOCODER_URL` to use another compatible reverse-geocoder, or use `--no-location` to disable the lookup.

The result is cached as normal Prospector provenance and is not used by the detector.

## Satellite context

Prospector uses Sentinel-2 Level-2A data via the Microsoft Planetary Computer STAC API. The service exposes a public STAC catalogue and Sentinel-2 data without requiring a user account for catalogue discovery; file access uses short-lived signed asset URLs. An optional `PC_SDK_SUBSCRIPTION_KEY` can be exported to use the Planetary Computer subscription-key rate-limit tier.

For each run, Prospector selects several relatively clear observations across time rather than trusting a single image. It calculates local NDVI anomaly support for each scene and uses the median support across selected scenes when ranking candidates. Sentinel-2 does not produce a visual base image in the interactive map. High-resolution Esri World Imagery provides the visual aerial/satellite base instead; Sentinel-2 remains an optional spectral support layer for detector scoring.

The spectral signal is deliberately **supporting evidence**, not a classifier. Cropmarks and vegetation responses may strengthen a candidate, but modern vegetation boundaries can also produce a response and are handled separately through mapped context. Sentinel-2 scenes are streamed as remote COG windows; Prospector stores the derived support raster and STAC provenance rather than copying complete Sentinel-2 scenes into every run.

## Detection model

The V0.4.4 detector is intentionally transparent and reproducible, with a local HE-trained terrain likelihood model that is evaluated before known-feature exclusion:

```text
LiDAR DTM
   |
   +--> multi-scale local relief
   +--> robust z-score anomaly field
   +--> cross-scale persistence
   +--> multi-scale linear/edge response + straight-line Hough seeding
   +--> annular/ring response
   +--> ridge/valley response + local texture/coherence
   +--> unsupervised Isolation Forest terrain novelty
   +--> generate candidates before Historic England exclusion, then validate known archaeology and remove matched areas from discovery output
   +--> connected candidate regions
            |
            +--> geometry / morphology
            +--> terrain-pattern channel scores
            +--> modern context penalty
            +--> satellite spectral support
            +--> Historic England LiDAR terrain-signature similarity
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
  discovery-score.tif
  terrain-novelty.tif
  ring-response.tif
  ridge-valley-response.tif

overlays/
  lidar-aim.png
  lidar-anomalies.png
  lidar-anomalies-aim.png
  high-resolution-imagery.png
  lidar-base.png
  lidar-anomalies-base.png

candidates/
  terrain-anomalies.geojson

reports/
  report.html

run.json
```

The exact set of files depends on whether external providers succeeded.

## Detection validation and learning

From v0.4.4, Prospector deliberately detects terrain before applying the Historic England exclusion mask. The known Historic England geometries are used as positive reference examples from the current LiDAR DTM, and the report measures how many known features the detector rediscovers. Those matched features are then removed from the final unknown-candidate set.

This gives each run a useful recall metric such as `43 / 51 known features validated` instead of silently hiding known archaeology before detection. Candidate similarity is based on LiDAR-derived terrain signatures (shape, multi-scale relief and morphology channels), not on the rendered PNG. The current model is label-guided within the study area rather than a general archaeological classifier; reviewed false positives and missed HE features should become the next training corpus.

## Install

Prospector is intended to run from the project's local `.venv`. The release includes an installer that always invokes the virtual environment's interpreter, so the console script is placed in `.venv/bin/` rather than the user Python installation.

```bash
./install.sh
source .venv/bin/activate
prospector --version
```

For a fully explicit install without activating the environment:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-user -e '.[all]'
.venv/bin/prospector --version
```

Do not use a system/user-site `pip` to install Prospector. The important invariant is that these two commands agree:

```bash
.venv/bin/python -m pip -V
.venv/bin/prospector --version
```

## Analyse

```bash
prospector analyse --latitude 50.8657 --longitude -0.2405
```

Detection sensitivity is now a 1–10 scale:

```bash
prospector analyse --latitude 50.8657 --longitude -0.2405 --sensitivity 1
prospector analyse --latitude 50.8657 --longitude -0.2405 --sensitivity 5
prospector analyse --latitude 50.8657 --longitude -0.2405 --sensitivity 10
```

`5` is the default balanced research setting: conservative enough to keep review practical, but broad enough to seed candidates from several independent terrain-pattern channels. `1` is deliberately selective; `10` is the exploratory “show me anything plausible” setting and can produce many false positives. `--workers 0` uses a conservative automatic detector thread count; a positive value overrides it.

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

V0.4.4 is deliberately a hybrid discovery system rather than an opaque model. Each candidate records the terrain signals that produced it. Human-reviewed candidate outcomes should eventually become the project's most valuable training data; that corpus can support a future supervised classifier without removing the transparent detector.

### Fast OS context

See **OS Data Hub API signup** above for the account, API project and environment-variable setup. `OS_DATAHUB_API_KEY` is the preferred variable; `OS_API_KEY` remains a legacy fallback.

## V0.5 training stack

V0.5 introduces an optional local learning stack for building an archaeological training corpus from Historic England AIM and LiDAR.

The stack is intentionally separate from the lightweight coordinate-driven analysis path. The existing analysis command continues to work without a training database or trained model. When `project/models/current.joblib` exists, `analyse` automatically uses it to add a persisted archaeology-model score to candidates.

The training stack uses:

- PostgreSQL + PostGIS for the spatial archaeology catalogue and model registry.
- Historic England AIM ingestion in bounded EPSG:27700 chunks. The current public service exposes `Detailed_Mapping`, `Monument_Extents` and `Project_Area`; all three are ingested, while `Detailed_Mapping` supplies positive archaeology examples.
- NLS/Environment Agency DTM tiles for LiDAR training data. Each known feature is represented at multiple physical scales and rotation augmentations so the model is not tied to one orientation or a single monument size.
- A grouped hold-out evaluation split so rotated/augmented copies of one monument cannot leak into both training and validation.
- A local `StandardScaler -> PCA -> ExtraTreesClassifier` model stored under `project/models/`, with a `current.joblib` pointer and JSON metadata.

Historic England's Aerial Investigation Mapping data is available as spatial download data and is updated periodically; the current [Historic England data-download page](https://historicengland.org.uk/listing/the-list/data-downloads) records the AIM dataset as last updated 25 June 2026. The public [HE AIM ArcGIS service](https://services-eu1.arcgis.com/ZOdPfBS3aqqDYPUQ/ArcGIS/rest/services/HE_AIM_data/FeatureServer) exposes the service used by Prospector. 

### Start the training stack

Create a local `.env` from the example and start PostGIS:

```bash
cp .env.example .env
docker compose up -d db
```

Compose waits for the PostGIS health check before starting the dependent application/trainer services. citeturn555997search0

Then initialise the schema:

```bash
docker compose run --rm trainer prospector train init
```

Ingest HE in chunks. For example, the 25000 m chunking below is deliberately bounded so the national corpus can be built incrementally:

```bash
docker compose run --rm trainer prospector train ingest-he \
  --xmin 450000 --ymin 50000 --xmax 550000 --ymax 150000 \
  --tile-size 25000
```

Build LiDAR-backed training examples. `--download-lidar` uses the same NLS/Environment Agency WCS provider as ordinary Prospector analysis, and the downloaded DTM chunks are cached under `project/training/lidar/`:

```bash
docker compose run --rm trainer prospector train build-dataset \
  --download-lidar \
  --xmin 450000 --ymin 50000 --xmax 550000 --ymax 150000 \
  --tile-size 1000
```

For iterative development, limit the number of tiles or features first:

```bash
docker compose run --rm trainer prospector train build-dataset \
  --download-lidar \
  --xmin 520000 --ymin 105000 --xmax 525000 --ymax 110000 \
  --tile-size 1000 --max-tiles 25 --max-features 100
```

Fit and register a model:

```bash
docker compose run --rm trainer prospector train fit \
  --dataset project/training/datasets/<dataset>/training-dataset.npz
```

Inspect the corpus and model registry:

```bash
docker compose run --rm trainer prospector train status
```

The model is deliberately treated as a research classifier, not proof of an archaeological identification. The next learning phase should add manually reviewed hard negatives, held-out geographic regions, and more complete monument-type labels before a V1.0 claim of broad accuracy.
