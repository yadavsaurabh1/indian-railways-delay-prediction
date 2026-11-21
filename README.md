# Indian Railways delay prediction

Forecasts the delay at every remaining stop of a running Indian Railways train. Each stop gets a 10th, 50th and 90th percentile, and the live predictor turns these into per-stop arrival and departure times.

This was a Minor Project for the Minor in Applied Machine Learning at the School of Electronics Engineering, Kalinga Institute of Industrial Technology (Deemed to be University). It was done under the guidance of Prof. Pravin Kumar Samanta, Assistant Professor (II), School of Electronics Engineering.

Authors: Saurabh Kumar Yadav, Aryan Vora, Ritika Rajak.

## Data

Training uses [`yadavsaurabh/indian-railways-runs`](https://huggingface.co/datasets/yadavsaurabh/indian-railways-runs) on Hugging Face. It holds 204,074,293 stop records for 12,935 trains and 9,701 stations, from January 2020 to September 2026. We collected it ourselves from the public whereismytrain.in `live_status` cache, using the harvester in [`yadavsaurabh1/train-data`](https://github.com/yadavsaurabh1/train-data).

## Results

The test set covers March to August 2026, 17.4M predicted stops. MAE is in minutes and is the error of the median forecast. Coverage is the share of actual delays that fall inside the 10th–90th percentile range, with 0.80 as the target.

| Model | MAE | 10–90 coverage |
|---|---|---|
| Persistence (current delay carried forward) | 28.05 | – |
| Current delay, or historical delay when unknown | 22.26 | – |
| Route transformer, 7.8M parameters | **14.69** | 0.785 |
| Route transformer + stop-to-stop traffic tokens, 101.1M parameters (still training) | 14.83 | 0.799 |

## Layout

| Path | What it does |
|---|---|
| `src/ingest.py` | Converts the CSVs to typed Parquet, one file per train, then partitions by month |
| `src/common.py` | Cleaning rules shared by every stage. Delay is recomputed as actual minus scheduled time, because the source leaves delays under 2 min empty |
| `src/build_static.py`, `src/build_history.py` | Station and train vocabularies, rake-link pairs, station congestion, and each train's strictly causal 90-day history |
| `src/fetch_weather.py` | Daily weather per 0.5° × 0.625° cell from NASA POWER |
| `src/build_month.py` | Training samples: a journey cut at a random time *t*, plus the other trains around its next 6 halts as they were known at *t* |
| `src/build_segments.py` | Stop-to-stop traffic: every other train recently on each upcoming stretch |
| `src/pack.py`, `src/features.py` | Flat memory-mapped arrays, and the model inputs shared by training and live inference |
| `src/model.py`, `src/train.py` | Route transformer with quantile outputs, trained on Apple GPUs (MPS) |
| `src/ablate.py`, `src/lgbm_baseline.py` | Ablation of the other-train inputs, and a LightGBM benchmark |
| `src/live.py`, `src/predict.py` | Live client for the `live_status` / `live_station` endpoints, and the per-stop ETA predictor |
| `scripts/` | The job scripts used on the training machine, in the order the project used them |

## Running

The environment is Python 3.12 managed with [uv](https://docs.astral.sh/uv/). The following commands run inside `src/`:

```sh
uv sync
uv run python ingest.py all --workers 8      # after downloading the dataset into data/raw
uv run python build_static.py && uv run python build_history.py
uv run python build_month.py --skip-done && uv run python fetch_weather.py
uv run python build_segments.py && uv run python pack.py
uv run python train.py --smoke               # wiring check
uv run python train.py                       # baseline model
uv run python predict.py 12624 04-11-2026    # live per-stop forecast
```

The scripts in `scripts/` were run on the training machine, where this code lived in a folder called `src_v2/`. Change that path if you rerun them.

## License

[GPL-3.0](LICENSE)
