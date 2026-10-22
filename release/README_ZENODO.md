# SAFEX-D: A Paired, Annotated Dataset for Conflict Forecasting in Dense Urban Air Mobility under Degraded Surveillance

**Version 1.0.0 (first public release).** 984 one-hour episodes (3 zones, 22 classes, 2 modes), 437 642 721 records at 1 Hz, average 126 and peak 485 concurrent aircraft. Simulation with BlueSky/OpenAP; wind for the weather episodes comes from real ERA5 hourly data. No real flight tracks are included.

## Contents of this record

The prefix `safex2` in file names comes from the internal name of the generator code (second generation of the generator, never released before); it is not a dataset version. This record is version 1.0.0.

| File | What it holds |
|---|---|
| `safex2_nom_free.tar`, `safex2_nom_mvp.tar` | nominal episodes (NOM-{A,B,C}-{LOW,NORMAL,BUSY}), ASAS off / ASAS on (MVP) |
| `safex2_cont_free.tar`, `safex2_cont_mvp.tar` | contingency episodes (7 classes) |
| `safex2_deg_free.tar`, `safex2_deg_mvp.tar` | degradation episodes (GNSS-L1, GNSS-L2, GNSS-LOSS, COMM-PARTIAL, COMM-FULL, SENSOR-FAULT) |
| `metadata/splits.csv` | episode, class, family, zone, density, mode, traffic group, fold (the 5 grouped folds used in the paper) |
| `metadata/corpus_summary.csv`, `class_table.csv`, `fleet_by_zone.csv` | corpus statistics |
| `metadata/real_wind.csv` | ERA5 hourly wind table used by the CONT-WEATHER episodes |
| `metadata/_campaign_log.csv` | generation log (one row per episode) |
| `metadata/*.csv`, `*.json` (benchmarks, link and age analyses) | result tables behind the paper |
| `MANIFEST.csv`, `CHECKSUMS.sha256` | every file with its archive and size; SHA-256 of archives and metadata |

Each archive holds, per episode `<ep>` (for example `NOM-A-BUSY_free_0001`): `<ep>.parquet` (one row per aircraft per second), `<ep>_conflicts.parquet` (pairwise conflict intervals) and `<ep>_meta.json` (parameters, event log, counts).

## Key fields of `<ep>.parquet`
`ep_t` (s since the start of the logged hour), `uav`, `type`, `lat`, `lon`, `x_m`, `y_m`, `alt_m`, `gs_ms`, `trk_deg`, `vs_ms` (true state); `rep_x_m`, `rep_y_m`, `rep_alt_m`, `rep_age_s` (state reported to the traffic management layer, with GNSS noise, outages and bias); `phase`, `cooperative`, `failed`, `comm_out`, `gnss_lost`, `sensor_fault`, `resolving`, `in_conflict_cd`; `min_sep_h_m`, `severity` (paper thresholds), `severity_half` (half-scale thresholds), `dist_vp_m`, `wind_ms`.
Labels are computed inside the simulator at 5 Hz from the true positions and logged at 1 Hz. Event episodes share their traffic with the nominal episode of the same traffic key, so each event has a paired baseline.

## Reading one episode
```python
import pandas as pd
d = pd.read_parquet("NOM-A-BUSY_free_0001.parquet")
```

## Licence and citation
Data: CC BY 4.0. Code: see the GitHub repository (linked in this record). Cite the DOI of this record and the paper.
