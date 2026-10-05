# Changelog

## 0.5.0

- Added local archaeological training pipeline with PostgreSQL/PostGIS.
- Added chunked Historic England AIM ingestion for Detailed_Mapping, Monument_Extents and Project_Area.
- Added LiDAR-backed, multi-scale and rotation-augmented training examples.
- Added grouped hold-out training with StandardScaler + PCA + ExtraTreesClassifier.
- Added local model registry and `project/models/current.joblib` inference integration.
- Added `prospector train init|ingest-he|build-dataset|fit|status` commands.
- Added Docker Compose stack for database, analysis and trainer services.
- Preserved existing coordinate-driven analysis and report generation as the production path while the learned system matures.
