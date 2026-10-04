# Architecture

V0.3.5 remains a single-process V1 application. It does **not** introduce the planned V2 database/ML training architecture.

```text
CLI
 |
 +--> StudyArea / EPSG:27700 canonical grid
 |
 +--> Historic England AIM provider
 |
 +--> LiDAR provider
 |      |
 |      +--> DTM
 |      +--> locked V0.2.1 visualisation
 |      +--> multi-scale anomaly field
 |
 +--> modern context providers
 |      +--> OS OpenMap Local
 |      +--> OpenStreetMap supplement
 |
 +--> satellite provider
 |      +--> Sentinel-2 STAC discovery
 |      +--> multi-scene NDVI support
 |
 +--> contextual ranker
 |      +--> cross-scale persistence
 |      +--> morphology
 |      +--> modernity penalty
 |      +--> satellite support
 |      +--> HE geometry similarity
 |
 +--> candidates / GeoJSON / HTML
```

## Canonical geospatial grid

The LiDAR raster is the authoritative analysis grid. All contextual raster outputs are reprojected onto:

- CRS: EPSG:27700
- exact LiDAR width/height
- exact LiDAR affine transform
- exact LiDAR bounds

The existing generated PNG overlays retain the same V0.2.1 renderer. The PNG terrain renderer remains frozen; V0.3.5 fixes the HTML/SVG overlay alignment separately by matching the SVG to matplotlib’s equal-aspect data viewport.

## Candidate scoring

V0.3.5 uses a transparent weighted rank:

```text
44% LiDAR strength
20% cross-scale persistence
18% morphology
10% satellite spectral support
 8% Historic England geometry similarity

then apply a modernity penalty
```

The score is deliberately not described as a probability.

## Context semantics

Modern features are soft context rather than universal exclusion. Buildings and roads receive the strongest penalty, while paths and boundaries remain softer because ancient features can survive beneath or beside modern infrastructure.

Historic England AIM `Detailed_Mapping` remains the known-archaeology reference and exclusion layer. Monument extents and project areas are preserved as contextual display layers, not treated as individual mapped features. The provider fully paginates all three vector layers before analysis. The absence of an AIM record is **unknown**, not negative training evidence.

## V2 boundary

A future learned model can consume V0.3.5 candidate outputs, human decisions and source provenance as training material. The current implementation deliberately avoids introducing PostGIS, pgvector, MLflow or a persistent training database until the deterministic feature/ranking layer has demonstrated useful behaviour.
