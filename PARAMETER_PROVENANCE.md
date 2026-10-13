# SAFEX-D (version 1.0.0): parameter provenance table

Status codes: **S** = sourced (URL given), **P** = partially sourced / sources disagree, **A** = assumption (declare in the paper, cover with a sensitivity analysis), **M** = modelling choice (justified in the text).
Only the flight envelope (speed, climb rate, turn behaviour) affects conflict geometry. Rotor power, battery and empty weight are carried by the performance model but do **not** change who conflicts with whom; they are listed for completeness.

## 1. Vehicles

| Vehicle (type code) | Parameter | Value | Status | Source / note |
|---|---|---|---|---|
| EHang EH216-S (`EH216`) | cruise speed | 28 m/s | S | 100 km/h = 27.8 m/s — https://www.evtol.news/ehang-216 |
| | max speed (envelope) | 36 m/s | S | 130 km/h = 36.1 m/s — evtol.news; Forecast International UAM Snapshot (http://flightplan.forecastinternational.com/2024/05/02/uam-snapshot-ehang-eh216-s/) |
| | motors / propellers | 16 | S | evtol.news; Forecast International |
| | MTOW | 600 kg | P | Forecast International gives **620 kg**; evtol.news gives no MTOW. Update to 620 kg (no effect on conflict geometry). |
| | range | 35 km | P | evtol.news 35 km; Forecast International 30 km. |
| | ceiling | 3000 m | S | evtol.news (max altitude 3,000 m) |
| | climb/descent rate | ±5 m/s | A | no public value found |
| | OEW 360 kg, motor power 15 kW | — | A | not published; not used by conflict logic |
| Generic medium eVTOL (`AAVM`) | all | cruise 45 m/s, MTOW 1800 kg | A | **No source.** Label it "generic medium eVTOL, illustrative" in the conference paper. For the journal replace it with a documented type (below). |
| Candidate replacement: Volocopter VoloCity | cruise / max speed | 48 kn (90 km/h = 25 m/s) / 54 kn (100 km/h) | S | Volocopter 2025 spec sheet: https://www.volocopter.com/cdn/assets/vnrac6vfvrab/3lVLdBP4Wmbv4e8hipNS1V/2adb898d09dbdd784183c26bd8bac960/2025_SpecSheet_VoloCit.pdf |
| | MTOM / rotors / range / seats | ≈1000 kg / 18 / ≈20 km / 2 | S | same sheet |
| sUAS types (`M600, Amzn, Mnet, Phan4, M100, M200, Mavic, Horsefly`) | mass, envelope (v_max, vs, h_max) | BlueSky OpenAP rotor database | P | Values come from the BlueSky/OpenAP aircraft file shipped with the simulator (e.g. Mavic Pro v_max 18 m/s, Phantom 4 20 m/s, Matrice 600 18 m/s, Matternet 11 m/s). Cite OpenAP/BlueSky; verify each against the manufacturer datasheet before the journal version. |
| | cruise speed per type | 9–18 m/s | M | Chosen **inside** each type's envelope (e.g. Mavic 14 vs v_max 18). Not a published operating speed. Cover with ±20 % sensitivity. |
| Eurocopter EC135 (`EC35`) | non-cooperative intruder, cruise 50 m/s | | M | OpenAP rotor entry (v_max 71.95 m/s); used only for the intruder scenario. |

## 2. Fleet and traffic

| Item | Value | Status | Note |
|---|---|---|---|
| eVTOL share A / B / C | 3 % / 30 % / 1 % | A | No public fleet statistics found. Replace by a sweep (e.g. 0, 3, 10, 30, 50 %) in the journal version. Context: Shenzhen plans >1,200 take-off/landing platforms by 2026, 249 built by mid-2024 (https://www.urbanairmobilitynews.com/new-city-projects/shenzhen-announces-plans-for-over-1200-drone-evtol-take-off-and-landing-platforms/); the article gives **no flight volumes**. |
| Vertiports | 12 per zone | A | Order of magnitude consistent with the platform count above; positions are synthetic. |
| Arrival rates | 520 / 1950 / 3900 per h | A | No public figure for Shenzhen. Present as density regimes, not predictions. |
| Zones | Futian CBD / Bao'an airport / Qianhai | M | "Inspired by" Shenzhen; not a reconstruction of real airspace. |

## 3. Safety thresholds and sensing

| Item | Value | Status | Note |
|---|---|---|---|
| LoWC / NMAC / MAC / vertical (profile `paper`) | 496.8 m / 152.4 m / 10 m / 30.48 m | M | Thresholds defined for SAFEX-D from the standards ; not taken from a released earlier version. `half` profile = sensitivity. |
| ASAS protected zone | ≈185 m (0.1 nm) | M | Smaller than the LoWC threshold; reported as a limitation. A matched-zone mode is planned for the journal version. |
| GNSS noise σ (L1 / L2) | 9 m / 18 m, AR(1) | A | Degradation levels, not measurements of a specific receiver. |
| Comm outage effect on ASAS | not modelled | M | Applied to the reported (UTM-visible) layer only. |
| Wind (CONT-WEATHER) | ERA5 hourly, 100 m wind ≥ 8 m/s | S | ERA5 via Open-Meteo archive; hour chosen by a rule fixed in advance. |
| Logging rate | 1 Hz (internal step 0.2 s) | M | Justified in the paper: ground truth at 5 Hz, release at the ~1 Hz rate of surveillance feeds. |

## 4. What to do with it
1. Conference paper: copy the table (shortened) as "Assumptions and sources". Keep `AAVM` labelled illustrative.
2. Journal version: (a) replace `AAVM` by VoloCity, (b) set EH216 MTOW to 620 kg, (c) verify the sUAS datasheets, (d) add ±20 % speed sensitivity and an eVTOL-share sweep, (e) regenerate zone B, then re-run the benchmark.
