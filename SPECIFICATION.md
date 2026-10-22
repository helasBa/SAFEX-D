# SAFEX-D (version 1.0.0): specification and run instructions

Data: https://doi.org/10.5281/zenodo.23350136.

## Design principles
* Ground truth is computed **inside the simulator at 5 Hz** from true positions and logged at 1 Hz as `severity` / `severity_half`; forecasting inputs never contain label-derived quantities.
* Landed UAVs are **deleted at touchdown** (no ghost records).
* 1 Hz logging, simulation step 0.2 s.
* True state **and** reported (UTM-visible) state with AR(1) GNSS noise, communication outage windows, GNSS loss and altitude bias; two threshold profiles (`paper`, `half`).
* Events are applied **physically** at `t_event` and logged in `meta["event_log"]`; each is verified by a counterfactual against the nominal episode with the same traffic key.
* Explicit `EH216` and `AAVM` types added by `patch_types.py` (**approximate / illustrative parameters, declare as assumptions**).
* Every scenario is run **twice**: `free` (ASAS off) and `mvp` (ASAS on, MVP resolution), with the same traffic.

## Taxonomy
* 9 nominal: `NOM-{A,B,C}-{LOW,NORMAL,BUSY}` (arrivals/h per zone: 520 / 1950 / 3900; `DENSE`=6500 is configured but not in the campaign).
* 7 contingency: VEHICLE-FAILURE, ALL-LAND, VERTIPORT-CLOSURE, WEATHER, NON-COOPERATIVE, INTRUDER, PASSENGER-DISTRESS.
* 6 degradation: GNSS-L1, GNSS-L2, GNSS-LOSS, COMM-PARTIAL, COMM-FULL, SENSOR-FAULT.
* Episode = 3600 s logged after a 600 s warm-up; event at t = 1800 s.
* Event episodes share `traffic_key = zone-density-idx` with `NOM-zone-density` of the same idx → exact paired design.

## Run (Windows, from a terminal)
```
python -m venv bsenv
bsenv\Scripts\activate
pip install bluesky-simulator pandas pyarrow numpy scipy
python safex2_campaign.py --preset pilot --out D:\safex2\pilot --workers 4
python safex2_verify.py D:\safex2\pilot -o D:\safex2\pilot_verify.csv
python safex2_campaign.py --preset core --out D:\safex2\data --workers 8   # 984 episodes
python safex2_campaign.py --preset full --out D:\safex2\data --workers 8   # 1606 (adds seeds, resumes)
```
Timing measured here (2 vCPU): one 3600 s BUSY episode (≈225 concurrent UAVs) = 111 s, 34 MB. Estimate for `core`: ≈ 25–30 CPU-hours, ≈ 20–25 GB.
Re-running the same command resumes; an episode counts as done when its `_meta.json` exists.

## Real-data anchoring (option B)
* **Weather** — `real/b1_fetch_real_wind.py` downloads ERA5 hourly wind (Open-Meteo archive API, no key, non-commercial) at the three zone centres. `CONT-WEATHER` episodes then draw one real hour
  (rule fixed in advance: 100 m wind >= 8 m/s, else the windiest 5 % of hours; seeded, identical in `free` and `mvp`) and ramp the wind to that speed/direction over 120 s. The chosen hour is stored in `meta["event_log"][0]["wind_source"]`.
  Without `--wind-csv` the generator falls back to the synthetic uniform(8, 14) m/s draw.
* **Kinematic plausibility** — `real/b2_real_anchor.py` compares simulated speed / climb-rate / turn-rate distributions with real ADS-B low-altitude traffic you export from OpenSky (account required).
  Nothing is trained on the real data. Known limitation to state in the paper: nominal simulated flight is idealised (constant cruise speed per type, straight legs; ASAS-on episodes add manoeuvres).
