# Roadmap

## V0.5 — learning foundation

- Ingest large, spatially chunked Historic England AIM datasets.
- Store source geometry and training provenance in PostGIS.
- Build LiDAR patches directly from elevation rasters.
- Make training examples robust to rotation and scale.
- Train a persisted local archaeology/background classifier.
- Score Prospector candidates with the persisted model when available.
- Preserve the existing HTML report and visual LiDAR overlays.

## V0.6 — archaeological object learning

- Expand monument-type labels and type-specific models.
- Add hard-negative mining from reviewed false positives.
- Add coarse model-driven candidate generation, not just candidate ranking.
- Add blind geographic validation across held-out projects/regions.
- Add model comparison and promotion gates.
- Add human review labels back into the training corpus.

## V0.7+ — discovery system

- Train on substantially larger national coverage.
- Add robust unknown-feature discovery and calibrated confidence.
- Add learned similarity search against known monument populations.
- Incorporate historical aerial imagery as a separate evidence modality.
- Add landscape-level relational features between candidate objects.

## V1.0 goal

V1.0 should not be declared until Prospector can demonstrate reproducible held-out geographic validation, strong recovery of known archaeology, controlled false-positive rates, and useful discovery performance on features not present in the training labels.
