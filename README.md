# SAFEX-D: code (version 1.0.0)

Generators, verification and analysis scripts for **SAFEX-D**, a simulated urban air mobility traffic corpus for conflict prediction with paired true and reported state.

- **Data:** https://doi.org/10.5281/zenodo.23350136 (984 episodes, 437 642 721 records, CC BY 4.0). The data are what you need to reproduce the paper's results: runs with automatic resolution (mode `mvp`) are reproducible statistically, not bit by bit.

The `safex2` prefix in script and data file names is the internal name of the generator code, not a version number. This is the first public release (v1.0.0).

## Layout

| Path | Content |
|---|---|
| `safex2_gen.py`, `safex2_campaign.py`, `patch_types.py` | episode generator (BlueSky + OpenAP) and campaign runner; `patch_types.py` adds the eVTOL types EH216 and AAVM |
| `safex2_verify.py` | per-episode checks (degradation effects, paired baselines) |
| `analysis/` | the pipeline a1 to a18 (corpus summary, task building, benchmark, foundation models, leakage audit, link emulation, figures, tables); run order in `analysis/README_analysis.md` |
| `real/` | ERA5 wind download (`b1_fetch_real_wind.py`) and a plausibility check against ADS-B tracks you export yourself (`b2_real_anchor.py`) |
| `tables_*.tex` | LaTeX tables as used in the paper |
| `SPECIFICATION.md`, `PARAMETER_PROVENANCE.md` | specification, declared assumptions, source of every parameter |

## Quick start (reproduce tables from the released data)

```
# 1. download the six archives and metadata from the Zenodo record, extract the .tar files into one folder DATA
# 2. install
pip install -r requirements-analysis.txt
# 3. corpus summary and folds, task tables, benchmark
cd analysis
python a1_corpus_summary.py DATA -o OUT
python a2_build_task.py DATA -o FEATS
python a3_benchmark.py FEATS --splits OUT\splits.csv -o OUT\benchmark
python a6_make_tables.py OUT
```

`metadata/splits.csv` in the Zenodo record already contains the folds, so a1 is only needed to regenerate the summary tables. Generating the corpus again needs BlueSky and about 25 to 30 CPU-hours (`SPECIFICATION.md`). The episodes were generated in a separate virtual environment, pinned in `requirements-generation.txt` (Python 3.13); the analysis scripts ran in the environment pinned in `requirements-analysis.txt`. The two environments have different numpy, pandas, pyarrow and openap versions; the analysis scripts do not use openap.

## Declared limits

Simulation with real ERA5 wind only. eVTOL parameters are approximate. Communication outages affect the reported-state layer, not the onboard avoidance logic. The surveillance link used in `a14` and `a15` is an application-level emulation (fixed delay, update period, independent packet loss), not a channel model.

## Licence

Code: MIT. Data: CC BY 4.0. BlueSky (GPL-3.0) is not redistributed here; cite it and give the version you used. If you redistribute modified BlueSky code, the GPL applies to it.
