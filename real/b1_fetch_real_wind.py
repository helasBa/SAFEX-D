"""Real wind (ERA5 reanalysis, served by Open-Meteo's free historical API) at the three SAFEX-D zone centres.
  python b1_fetch_real_wind.py -o real_wind.csv [--start 2023-01-01 --end 2023-12-31]
No account or key is needed for non-commercial research use; cite Open-Meteo and ERA5 (Copernicus / ECMWF) in the paper.
Writes one row per zone and hour:  zone, time (UTC), wind_speed_10m_ms, wind_speed_100m_ms, wind_direction_100m_deg (direction the wind comes FROM), wind_gusts_10m_ms.
Then regenerate the weather episodes with this table:  python safex2_campaign.py --preset core --out DATA --wind-csv real_wind.csv  (after deleting CONT-WEATHER* from DATA)."""
import argparse, json, sys, time, urllib.parse, urllib.request
import pandas as pd

ZONES = {"A": (22.54, 114.06), "B": (22.60, 113.88), "C": (22.50, 114.03)}     # must match safex2_gen.ZONES
VARS = ["wind_speed_10m", "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m"]

def fetch(lat, lon, start, end, tries=4):
    q = urllib.parse.urlencode(dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=",".join(VARS), wind_speed_unit="ms", timezone="UTC", models="era5"))
    for k in range(tries):
        try:
            with urllib.request.urlopen("https://archive-api.open-meteo.com/v1/archive?" + q, timeout=60) as r: return json.load(r)
        except Exception as e:
            print("  retry", k + 1, repr(e)[:120], file=sys.stderr); time.sleep(5 * (k + 1))
    raise SystemExit("Open-Meteo request failed; check your connection and the dates")

def to_frame(zone, js):
    h = js["hourly"]; d = pd.DataFrame({"zone": zone, "time": h["time"], "wind_speed_10m_ms": h["wind_speed_10m"], "wind_speed_100m_ms": h["wind_speed_100m"],
                                        "wind_direction_100m_deg": h["wind_direction_100m"], "wind_gusts_10m_ms": h["wind_gusts_10m"]})
    return d.dropna().reset_index(drop=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-o", default="real_wind.csv"); ap.add_argument("--start", default="2023-01-01"); ap.add_argument("--end", default="2023-12-31"); a = ap.parse_args()
    D = pd.concat([to_frame(z, fetch(la, lo, a.start, a.end)) for z, (la, lo) in ZONES.items()], ignore_index=True); D.to_csv(a.o, index=False)
    print(len(D), "hourly rows ->", a.o)
    print(D.groupby("zone").wind_speed_100m_ms.describe().round(2).to_string())
    print("hours with 100 m wind >= 8 m/s:", D[D.wind_speed_100m_ms >= 8].groupby("zone").size().to_dict())
