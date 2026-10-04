# Roadmap

## V1.0 — current line

- Improve candidate ranking using multiple LiDAR scales.
- Add morphology-aware scoring.
- Add OS and OpenStreetMap modern context.
- Add satellite contextual evidence.
- Preserve the proven LiDAR visualisation renderer.
- Improve report explanations and candidate provenance.
- Build a repeatable corpus of human-reviewed false positives and useful candidates.

## V2.0 — planned learned system

- ingest the full Historic England Aerial Investigation Mapping archive where licensing permits
- introduce a persistent spatial database
- create labelled positive archaeology and hard-negative modern/agricultural training sets
- train a segmentation/candidate-generation model
- train a multimodal candidate ranker
- add embedding/vector similarity search
- add active learning from human review
- add blind spatial validation by held-out regions
- integrate historic aerial photography as another evidence modality

## Overlay alignment

The generated PNG overlays are already the stable visualisation baseline. The HTML/SVG alignment bug is a separate issue and should be solved without changing the PNG renderer or detector inputs.
