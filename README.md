# Ocean Embed

Ocean Embed is a prototype for spatiotemporal ocean-surface wind forecasting from NetCDF data.

## Run locally

From the repository root:

```bash
PYTHONPATH=. python3 scripts/inspect_data.py
PYTHONPATH=. python3 scripts/smoke_test.py
PYTHONPATH=. python3 scripts/run_experiment.py
PYTHONPATH=. python3 training/train_convlstm.py --epochs 20
PYTHONPATH=. python3 training/train_vit.py --epochs 20
PYTHONPATH=. python3 training/evaluate_checkpoints.py
PYTHONPATH=. pytest -q
PYTHONPATH=. uvicorn ocean_embed.api.main:app --reload
```

The API is available at `http://localhost:8000`; the minimal dashboard is [frontend/index.html](frontend/index.html).

Place each month's files in a subfolder under `dataset/` (for example `dataset/jan_20`, `dataset/feb_20`). The scripts recursively discover only files containing both wind components, validate compatible grids, concatenate them chronologically, and remove duplicate timestamps. Run `training/train_convlstm.py` and `training/train_vit.py` separately; each uses valid-pixel masked loss and writes its own best checkpoint. Use `--epochs`, `--batch-size`, `--learning-rate`, or `--data` to override defaults. `--data` can point to another dataset root or a single wind file.

To test the saved models directly on the held-out test period, run `PYTHONPATH=. python3 training/evaluate_checkpoints.py`. This loads both checkpoints and prints U/V MAE, speed MAE, direction error, RMSE, and valid-pixel coverage.

## Current MVP dataset

The ASCAT wind file is the only current dataset directly compatible with the first experiment. It contains 31 daily observations on a 68×80, 0.25° grid with `eastward_wind` and `northward_wind`. Approximately 74.7% of wind pixels are missing, so metrics use valid-pixel masks. The generated registry is written to `outputs/dataset_registry.json` and the latest test metrics to `outputs/metrics/experiment.json`.

This is a limited-data prototype experiment. The short time series is suitable for pipeline verification, not for strong model-quality conclusions.

## Architecture

- `ocean_embed/preprocessing`: xarray loading, compatibility validation, normalization, and temporal sequences.
- `ocean_embed/models`: persistence, ConvLSTM, and spatiotemporal ViT.
- `ocean_embed/training`: masked wind metrics and circular direction error.
- `scripts`: inspection, smoke test, and a small real-data experiment.
- `ocean_embed/api`: dataset inventory, health, metrics, and prediction contract.

See [OCEAN_EMBED_PROJECT.md](OCEAN_EMBED_PROJECT.md) for the full blueprint and remaining production-hardening work.
