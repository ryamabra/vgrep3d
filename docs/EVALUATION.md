# Evaluation Guide

Support count alone is not a localization metric: a method can select most of a scene and still contain the target. Evaluate both whether a query returns a result and how tightly it localizes the intended object.

## Suggested metrics

- **Found rate:** fraction of prompts that return a localization.
- **Support fraction:** selected Gaussians divided by total scene Gaussians. Lower is usually tighter, but it is not meaningful without checking recall.
- **Centroid error:** Euclidean distance from the predicted centroid to an annotated object center.
- **3D IoU:** intersection-over-union between predicted and annotated boxes.
- **Prompt stability:** variation across paraphrases such as `red car`, `red vehicle`, and `the red sedan`.
- **View stability:** variation when the same scene is reconstructed from different frame subsets.

Report per-query values as well as aggregates. Keep failed queries in the dataset; dropping them inflates the apparent success rate.

## JSON Lines format

The summary utility accepts one JSON object per line. Only `found` is required. Include `prompt`, `num_gaussians`, thresholds, and scene metadata whenever available.

```json
{"scene":"drive_01","prompt":"red car","found":true,"num_gaussians":473054,"threshold":0.6}
{"scene":"drive_01","prompt":"stop sign","found":false,"threshold":0.7}
```

Run the summary from an editable install or with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python scripts/summarize_queries.py results.jsonl
PYTHONPATH=src python scripts/summarize_queries.py results.jsonl --json
```

## Known failure cases

- A high found rate with a large support fraction indicates scene-level matching, not successful object localization.
- Centroid error can look good for a loose box centered near the object; pair it with 3D IoU or support fraction.
- Comparisons across different Gaussian caps are invalid unless support is normalized by scene size.
- Grounding DINO backprojection results should be labeled as the hybrid baseline, not as feature-field results.
- Single-view or forward-only captures should be reported separately from orbit-style captures because view diversity materially changes the task.
