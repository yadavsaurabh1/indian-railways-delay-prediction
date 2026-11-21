"""Daily weather per 0.5 x 0.625 degree cell (the NASA POWER / MERRA-2 grid) for every
cell that contains a station, 2020-01-01 .. 2026-09-01.

Source: NASA POWER daily point API (free, no key). One request per cell.
Variables: T2M (mean temp C), T2M_MIN, T2MDEW (dew point), RH2M (%), PRECTOTCORR (mm/day),
WS2M (wind m/s at 2 m). Fog risk is high when T2M_MIN - T2MDEW is small, RH high and wind low.

Outputs: data/static/weather_cells.parquet (station_id -> cell), data/static/weather_daily.parquet
Resumable: raw JSON per cell cached under data/weather_raw/.
"""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl

from common import DATA, STATIC

DLAT, DLNG = 0.5, 0.625
PARAMS = ["T2M", "T2M_MIN", "T2MDEW", "RH2M", "PRECTOTCORR", "WS2M"]
START, END = "20200101", "20260901"
RAW = DATA / "weather_raw"
URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters={p}&community=RE"
       "&longitude={lng}&latitude={lat}&start={s}&end={e}&format=JSON")


def cell_of(lat: pl.Expr, lng: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    return ((lat / DLAT).round() * DLAT).round(4), ((lng / DLNG).round() * DLNG).round(4)


def fetch(lat: float, lng: float) -> str:
    out = RAW / f"{lat:.4f}_{lng:.4f}.json"
    if out.exists():
        return str(out)
    url = URL.format(p=",".join(PARAMS), lat=lat, lng=lng, s=START, e=END)
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "indian-railways-model/0.1"})
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read()
            json.loads(body)
            out.write_bytes(body)
            return str(out)
        except Exception as e:  # network hiccup / throttling
            wait = 10 * 2 ** attempt
            print(f"  retry {lat},{lng} in {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed {lat},{lng}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    st = pl.read_parquet(STATIC / "stations.parquet").drop_nulls(["lat", "lng"])
    clat, clng = cell_of(pl.col("lat"), pl.col("lng"))
    st = st.with_columns(clat.alias("clat"), clng.alias("clng"))
    st.select("station_id", "clat", "clng").write_parquet(STATIC / "weather_cells.parquet")
    cells = st.select("clat", "clng").unique().rows()
    print(f"[weather] {len(cells)} cells", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(fetch, a, b) for a, b in cells]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 50 == 0 or i == len(cells):
                print(f"[weather] {i}/{len(cells)} {time.time() - t0:.0f}s", flush=True)

    frames = []
    for a, b in cells:
        d = json.loads((RAW / f"{a:.4f}_{b:.4f}.json").read_text())["properties"]["parameter"]
        days = list(d[PARAMS[0]].keys())
        frames.append(pl.DataFrame({
            "clat": [a] * len(days), "clng": [b] * len(days),
            "day": pl.Series(days).str.strptime(pl.Date, "%Y%m%d"),
            **{p.lower(): [None if v is None or v <= -998 else float(v) for v in d[p].values()] for p in PARAMS},
        }))
    w = pl.concat(frames).with_columns(pl.col(pl.Float64).cast(pl.Float32))
    w.write_parquet(STATIC / "weather_daily.parquet")
    print(f"[weather] wrote {w.height:,} rows, nulls: {w.null_count().row(0)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
