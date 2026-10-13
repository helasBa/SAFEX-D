# SAFEX-D v2 — analysis pipeline (a1 → a6)

All scripts run on your PC, on the folder produced by `safex2_campaign.py` (call it `DATA` = `C:\these\safex2_data`).
Outputs go to `OUT` = `C:\these\safex2_out`. Every step is resumable and writes small files you can send back.

## 0. Install (once, in the same `bsenv`)
```
pip install --no-cache-dir scikit-learn
```
(For step 5 only: `pip install --no-cache-dir torch "chronos-forecasting>=2.0"` and the TimesFM install line in `a5_foundation_model.py`.)

## 1. Corpus summary and splits (~5 min)
```
python a1_corpus_summary.py C:\these\safex2_data -o C:\these\safex2_out --workers 6
```
Send back: `class_table.csv`, `splits.csv` (small).

## 2. Paired event effects (~5-10 min)
```
python a4_event_effect.py C:\these\safex2_data -o C:\these\safex2_out --workers 6
```
Send back: `event_effect_summary.csv`.

## 3. Build the forecasting task (~15-20 min, ~2-3 GB)
```
python a2_build_task.py C:\these\safex2_data -o C:\these\safex2_feats --workers 6
```

## 4. Benchmark (the long one: several hours, resumable — stop it any time, rerun the same command)
Quick first look (≈ 40 min):
```
python a3_benchmark.py C:\these\safex2_feats --splits C:\these\safex2_out\splits.csv -o C:\these\safex2_out\benchmark --targets paper_tac --horizons 30 --boot 40
```
Full (all targets and horizons; leave it running overnight):
```
python a3_benchmark.py C:\these\safex2_feats --splits C:\these\safex2_out\splits.csv -o C:\these\safex2_out\benchmark
```
Send back: `benchmark.json`.

## 5. Foundation models on your GPU
```
python a5_foundation_model.py C:\these\safex2_data --feats C:\these\safex2_feats --splits C:\these\safex2_out\splits.csv --model cv        -o C:\these\safex2_out\fm_cv
python a5_foundation_model.py C:\these\safex2_data --feats C:\these\safex2_feats --splits C:\these\safex2_out\splits.csv --model chronos2  --device cuda -o C:\these\safex2_out\fm_chronos2
python a5_foundation_model.py C:\these\safex2_data --feats C:\these\safex2_feats --splits C:\these\safex2_out\splits.csv --model timesfm25 --device cuda -o C:\these\safex2_out\fm_timesfm25
```
Run `--model cv` first: it must reproduce the harness baseline and needs no network. Send back the three `fm_*.json`.

## 6. LaTeX tables
```
python a6_make_tables.py C:\these\safex2_out
```
Produces `tables_v2.tex` (every number copied from the result files).

## Design notes (for the paper's method section)
* Inputs are the UTM-visible (reported) state only; no severity, separation or resolution column is ever an input.
* Targets are future events within 10/30/60 s for aircraft currently conflict-free; two threshold profiles (`paper`, `half`) and two levels (tactical = LoWC, imminent = NMAC).
* Splits: all episodes with the same traffic key (NOM and event episodes, free and mvp) share a fold; confidence intervals resample traffic groups.
* Protocols: grouped 5-fold CV, leave-one-zone-out, nominal→event transfer, free↔mvp transfer.
* Baseline: constant-velocity physics; learned models use HistGradientBoosting; the hybrid receives the physics score as a feature.
