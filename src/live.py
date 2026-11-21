"""Live inputs for prediction, from the whereismytrain.in cache.

live_status(train_no, start_date)  -> one running journey, in the same schema as
                                      data/staged/*.parquet (see ingest.py), so the
                                      model sees live data exactly as it saw history.
live_station(station_code)         -> trains due at a station over roughly the
                                      surrounding ±6 h, with their current delay.
                                      This is the "other trains on my route" input.

Mirrors the dataset rule: a stop counts as observed only when it has an
`updated_at`. The source fills in projected times for stops that never
reported, and those are blanked here exactly as they are in the dataset.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone, timedelta

import polars as pl

BASE = "https://whereismytrain.in/cache"
IST = timezone(timedelta(hours=5, minutes=30))
USER_AGENT = "indian-railways-model/0.1"
MIN_INTERVAL_S = 1.0  # be polite: at most one request per second
CACHE_TTL_S = 60.0

_last_request = 0.0
_cache: dict[str, tuple[float, dict]] = {}


def _get(path: str, **params) -> dict:
    global _last_request
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    hit = _cache.get(url)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    wait = MIN_INTERVAL_S - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    _last_request = time.time()
    _cache[url] = (time.time(), data)
    return data


def _from_epoch(tm) -> datetime | None:
    return datetime.fromtimestamp(tm, IST).replace(tzinfo=None) if tm else None


def _iso_ist(s: str | None) -> datetime | None:
    """ISO timestamp -> naive IST. Placeholders like 'Source'/'Destination' -> None."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).astimezone(IST).replace(tzinfo=None)
    except ValueError:
        return None


_HHMM = re.compile(r"^(-)?(\d+):(\d{2})$")


def delay_minutes(s: str | None) -> float | None:
    """'01:38' -> 98.0, 'RIGHT TIME' -> 0.0, '-00:05' -> -5.0, blank -> None."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if s.upper().startswith("RIGHT"):
        return 0.0
    m = _HHMM.match(s)
    if not m:
        return None
    v = int(m.group(2)) * 60 + int(m.group(3))
    return float(-v if m.group(1) else v)


def live_status(train_no: str, start_date: date | str) -> tuple[pl.DataFrame, dict]:
    """Returns (stops in staged-parquet schema, journey-level metadata)."""
    if isinstance(start_date, date):
        start_date = start_date.strftime("%d-%m-%Y")
    d = _get("live_status", date=start_date, train_no=str(train_no).zfill(5))
    jdate = datetime.strptime(d["start_date"], "%d-%m-%Y").date()
    rows = []
    for s in d.get("days_schedule", []):
        reported = s.get("updated_at") is not None
        rows.append({
            "date": jdate,
            "train_no": str(train_no).zfill(5),
            "train_name": d.get("train_name"),
            "train_type": d.get("train_type"),
            "seq": float(s["sno"]) if s.get("sno") is not None else None,
            "station_code": s.get("station_code"),
            "halts_here": 1 if s.get("stops") else 0,
            "sch_arr": _from_epoch(s.get("sch_arrival_tm")),
            "sch_dep": _from_epoch(s.get("sch_departure_tm")),
            "act_arr": _from_epoch(s.get("actual_arrival_tm")) if reported else None,
            "act_dep": _from_epoch(s.get("actual_departure_tm")) if reported else None,
            "delay_arrival_min": float(s["delay_in_arrival"]) if reported and s.get("delay_in_arrival") is not None else None,
            "delay_departure_min": float(s["delay_in_departure"]) if reported and s.get("delay_in_departure") is not None else None,
            "platform": s.get("platform") or None,
            "distance_km": s.get("distance"),
            "lat": s.get("lat"),
            "lng": s.get("lng"),
            "departed": 1 if s.get("departed") else 0,
            "reported": 1 if reported else 0,
            "reported_at": _iso_ist(s.get("updated_at")),
        })
    df = pl.DataFrame(rows, schema={
        "date": pl.Date, "train_no": pl.String, "train_name": pl.String, "train_type": pl.String,
        "seq": pl.Float32, "station_code": pl.String, "halts_here": pl.Int8,
        "sch_arr": pl.Datetime("us"), "sch_dep": pl.Datetime("us"),
        "act_arr": pl.Datetime("us"), "act_dep": pl.Datetime("us"),
        "delay_arrival_min": pl.Float32, "delay_departure_min": pl.Float32, "platform": pl.String,
        "distance_km": pl.Float64, "lat": pl.Float64, "lng": pl.Float64,
        "departed": pl.Int8, "reported": pl.Int8, "reported_at": pl.Datetime("us"),
    })
    meta = {
        "current_station": d.get("curStn"),
        "next_halt": (d.get("pitstop_next_to_curstn") or {}).get("station_code"),
        "last_update": _iso_ist(d.get("lastUpdateIsoDate")),
        "source_station": d.get("source_station"),
        "destination_station": d.get("destination_station"),
        "total_distance_km": d.get("distance"),
    }
    return df, meta


def live_station(station_code: str) -> pl.DataFrame:
    d = _get("live_station", station_code=station_code)
    rows = []
    for r in d.get("live_station_info", []):
        rows.append({
            "station_code": station_code,
            "train_no": str(r.get("train_no", "")).zfill(5),
            "train_type": r.get("train_type"),
            "start_date": datetime.strptime(r["startDate"], "%d %b %Y").date() if r.get("startDate") else None,
            "source_station": r.get("source_station"),
            "destination_station": r.get("destination_station"),
            "current_station": r.get("cur_stn"),
            "sch_arr": _iso_ist(r.get("scheduled_arrival")),
            "sch_dep": _iso_ist(r.get("scheduled_departure")),
            "delay_arrival_min": delay_minutes(r.get("delay_in_arrival")),
            "delay_departure_min": delay_minutes(r.get("delay_in_departure")),
            "sch_halt_min": float(r["schHalt"]) if str(r.get("schHalt", "")).isdigit() else None,
            "act_halt_min": float(r["actHalt"]) if str(r.get("actHalt", "")).isdigit() else None,
            "platform": r.get("platform") or None,
            "arrived": bool(r.get("arrived")),
            "departed": bool(r.get("departed")),
            "cancelled": bool(r.get("cancelled")),
            "diverted": bool(r.get("diverted")),
            "last_update": _iso_ist(r.get("last_update_isodate")),
        })
    df = pl.DataFrame(rows, schema={
        "station_code": pl.String, "train_no": pl.String, "train_type": pl.String, "start_date": pl.Date,
        "source_station": pl.String, "destination_station": pl.String, "current_station": pl.String,
        "sch_arr": pl.Datetime("us"), "sch_dep": pl.Datetime("us"),
        "delay_arrival_min": pl.Float32, "delay_departure_min": pl.Float32,
        "sch_halt_min": pl.Float32, "act_halt_min": pl.Float32, "platform": pl.String,
        "arrived": pl.Boolean, "departed": pl.Boolean, "cancelled": pl.Boolean, "diverted": pl.Boolean,
        "last_update": pl.Datetime("us"),
    })
    return df.with_columns(snapshot_at=pl.lit(_iso_ist(d.get("modified_at")), dtype=pl.Datetime("us")))


if __name__ == "__main__":
    import sys
    with pl.Config(tbl_cols=25, tbl_width_chars=220, tbl_rows=8):
        if len(sys.argv) == 3:
            df, meta = live_status(sys.argv[1], sys.argv[2])
            print(meta)
            print(df)
        else:
            print(live_station(sys.argv[1]))
