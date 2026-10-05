# Changelog

## 0.5.2

- Fixed the Docker geospatial/training image by installing required native runtime libraries for compiled wheels.
- Added explicit `libexpat1` validation for Fiona/GDAL and `libgomp1` validation for the scientific/ML stack.
- Made the Docker dependency smoke test verify both Python imports and native shared libraries.

## 0.5.1

- Fixed Docker training images installing Prospector without the geospatial extras required by LiDAR validation and dataset generation.
- Expanded the `all` extra explicitly instead of recursively referencing Prospector's own extras.
- Added Docker build-time imports to fail the image build if a required geospatial/training dependency is missing.
- Disabled Python user-site package discovery inside the Docker image.
- Added `notes.txt` as a concise training-stack cheatsheet.

## 0.5.0

- Added the local HE/LiDAR training stack, PostGIS catalogue, dataset generation, model training and model-backed inference.
