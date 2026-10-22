# Datasheet: SAFEX-D, version 1.0.0

Format follows "Datasheets for Datasets" (Gebru et al.). Fields in [brackets] are for the authors to fill.

## Motivation
- **Purpose.** Forecasting conflicts between aircraft in dense urban air mobility traffic, with the true state and the state reported to the traffic management layer kept side by side, so that forecasting quality can be measured under degraded surveillance.
- **Created by.** [authors, affiliation]. **Funding.** [funding statement].

## Composition
- 984 one-hour episodes (3600 s logged after a 600 s warm-up), 437 642 721 records at 1 Hz, three urban zones (A, B, C), 22 scenario classes, two modes per scenario: `free` (automatic resolution off) and `mvp` (MVP resolution on). Episodes sharing a traffic key use the same generated traffic.
- Nominal 360 episodes (180 per mode), contingency 336 (7 classes), degradation 288 (6 classes). Average 126 and peak 485 concurrent aircraft.
- Each record: true state, reported state (GNSS noise, outages, bias), phase and failure flags, pairwise separation, conflict severity under two threshold profiles (`paper`, `half`), distance to the nearest vertiport, wind. Pairwise conflict intervals and per-episode metadata are stored next to each episode.
- No personal data. No real flight tracks. Wind for the weather episodes is real ERA5 hourly data (Open-Meteo archive API).
- `metadata/splits.csv` gives the 5 grouped folds (180 traffic groups) used for all reported results.

## Collection process
- Generated with BlueSky and OpenAP using the scripts in the code repository. Labels are computed inside the simulator at 5 Hz from true positions and logged at 1 Hz. Events are applied physically at t = 1800 s and checked against the paired nominal episode.
- The eVTOL types EH216 and AAVM use approximate, illustrative performance parameters (see `PARAMETER_PROVENANCE.md` in the repository).

## Preprocessing
None beyond the simulator output. Derived tables used for the benchmarks (feature tables, emulated-link variants) are not included and can be regenerated with the repository scripts.

## Uses
- **Intended.** Benchmarking conflict forecasting methods (physics rules, learned models, foundation models), studying the effect of report age and link degradation, studying the effect of automatic resolution with paired runs.
- **Not intended.** Certifying a system, estimating real-world conflict rates (with the `paper` thresholds the share of rows with a loss of well-clear is high in dense zones, so base rates are not realistic), training operational systems without validation on real data.

## Known limits
Simulation only. Communication outages affect the reported-state layer and not the onboard avoidance logic. Nominal flight is idealised (constant cruise speed per type, straight legs). Fleet mix and vertiport layout are assumptions. Runs with automatic resolution are reproducible statistically, not bit by bit.

## Distribution and maintenance
Archived on Zenodo 10.5281/zenodo.23350136, CC BY 4.0. Code: [GitHub URL] (release v1.0.0). Contact: [email]. Corrections are published as new Zenodo versions.
