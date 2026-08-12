"""Real F1 team radio from the Hugging Face dataset `MikCil/f1-team-radio`.

Pulls one driver's radio clips for one Grand Prix, aligns each clip to the lap it
was actually sent on (matching the clip's UTC `message_timestamp` against FastF1's
absolute per-lap start times), and downloads the ones inside a lap window as
`clips/lap<N>.mp3`. The existing live pipeline then runs Whisper + wav2vec2 on
genuine driver radio, so the emotion/stress arc is real rather than synthetic.

The dataset is queried through the HF datasets-server `/filter` API, so we only
pull the handful of rows for the chosen driver+GP (not the full 2.5 GB), and each
clip's audio arrives as a short-lived signed mp3 URL that we download immediately.
"""
import os
import json
import time
import functools
import urllib.parse
import urllib.request
import concurrent.futures

DATASET = "MikCil/f1-team-radio"
SERVER = "https://datasets-server.huggingface.co"


def _filter(where, length=100, offset=0, retries=5):
    qs = urllib.parse.urlencode({
        "dataset": DATASET, "config": "default", "split": "train",
        "where": where, "length": length, "offset": offset,
    })
    url = SERVER + "/filter?" + qs
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:   # datasets-server 500s intermittently; retry with backoff
            last = e
            if attempt < retries - 1:
                time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"dataset filter failed after {retries} tries: {last}")


@functools.lru_cache(maxsize=1)
def _statistics():
    qs = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "train"})
    with urllib.request.urlopen(SERVER + "/statistics?" + qs, timeout=90) as r:
        return json.load(r)


def catalog():
    """Dataset taxonomy for the UI dropdowns, from the grand_prix value counts:
    {years: [...newest first], by_year: {year: [{name, dataset_gp, count}]}}.
    `name` is the GP with the leading year stripped (the FastF1 event name)."""
    import localset
    if localset.is_available():
        return localset.catalog()
    gp_freq = {}
    for s in _statistics().get("statistics", []):
        if s["column_name"] == "grand_prix":
            gp_freq = s["column_statistics"].get("frequencies") or {}
    by_year = {}
    for dataset_gp, count in gp_freq.items():
        parts = str(dataset_gp).split(" ", 1)          # "2020 Turkish Grand Prix"
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        year, name = parts[0], parts[1]
        by_year.setdefault(year, []).append(
            {"name": name, "dataset_gp": dataset_gp, "count": count})
    for year in by_year:
        by_year[year].sort(key=lambda g: g["name"])
    return {"years": sorted(by_year, reverse=True), "by_year": by_year}


@functools.lru_cache(maxsize=256)
def _gp_rows(dataset_gp, max_rows=600):
    """All rows for one GP (racing_number, driver_id, message_timestamp), fetched
    once and cached — both drivers_for_gp and sessions_for_gp read from this, so a
    GP is only paged from the slow HF /filter a single time (or read from the local
    parquet instantly when the dataset has been downloaded)."""
    import localset
    if localset.is_available():
        return localset.gp_rows(dataset_gp)
    where = f"\"grand_prix\"='{str(dataset_gp).replace(chr(39), chr(39) * 2)}'"
    rows, offset = [], 0
    while offset < max_rows:
        d = _filter(where, length=100, offset=offset)
        batch = [r["row"] for r in d.get("rows", [])]
        rows.extend(batch)
        total = d.get("num_rows_total", offset + len(batch))
        offset += len(batch)
        if not batch or offset >= total:
            break
    return rows


def drivers_for_gp(dataset_gp):
    """Distinct drivers who have radio at this GP, richest first:
    [{racing_number, abbr, driver_id, count}]. abbr is the standard 3-letter code
    encoded in driver_id (e.g. LEWHAM01 -> HAM)."""
    seen = {}
    for row in _gp_rows(dataset_gp):
        num = str(row.get("racing_number"))
        did = row.get("driver_id") or ""
        abbr = did[3:6].upper() if len(did) >= 6 else did.upper()
        e = seen.setdefault(num, {"racing_number": num, "abbr": abbr,
                                  "driver_id": did, "count": 0})
        e["count"] += 1
    return sorted(seen.values(), key=lambda e: -e["count"])


# FastF1 session name -> the code get_session accepts (and we pass to import/analyze)
_SESSION_CODE = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Qualifying": "Q", "Sprint": "S", "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SS", "Race": "R",
}


@functools.lru_cache(maxsize=64)
def sessions_for_gp(year, gp_name, dataset_gp):
    """Which sessions of this GP actually have radio, in schedule order:
    [{code, name, count}]. Each clip's UTC timestamp is bucketed into the session
    whose start time most recently precedes it (from FastF1's event schedule)."""
    import fastf1
    import pandas as pd
    os.makedirs(".fastf1cache", exist_ok=True)
    fastf1.Cache.enable_cache(".fastf1cache")

    ev = fastf1.get_event(int(year), gp_name)
    schedule = []   # (name, code, start_naive_utc), in session order
    for i in range(1, 6):
        name = ev.get(f"Session{i}")
        dt = ev.get(f"Session{i}DateUtc")
        if name and pd.notna(dt):
            start = pd.Timestamp(dt)
            start = start.replace(tzinfo=None) if start.tzinfo else start
            schedule.append((str(name), _SESSION_CODE.get(str(name), str(name)), start))
    if not schedule:
        return [{"code": "R", "name": "Race", "count": 0}]

    counts = {}
    for row in _gp_rows(dataset_gp):
        t = pd.Timestamp(row["message_timestamp"])
        t = t.replace(tzinfo=None) if t.tzinfo else t
        best = None
        for name, code, start in schedule:
            if start <= t:
                best = code
        if best:
            counts[best] = counts.get(best, 0) + 1

    return [{"code": code, "name": name, "count": counts.get(code, 0)}
            for name, code, _ in schedule if counts.get(code, 0) > 0]


@functools.lru_cache(maxsize=64)
def teams_for_gp(year, gp_name, session, dataset_gp):
    """[{team, drivers:[...], count}] for this GP — the dataset's radio-having drivers
    grouped into constructors via FastF1's driver->team mapping. Richest team first."""
    drivers = {d["racing_number"]: d for d in drivers_for_gp(dataset_gp)}
    import fastf1
    os.makedirs(".fastf1cache", exist_ok=True)
    fastf1.Cache.enable_cache(".fastf1cache")
    ses = fastf1.get_session(int(year), gp_name, session)
    ses.load(telemetry=False, weather=False, messages=False)
    pairs = ses.laps[["DriverNumber", "Team"]].dropna().drop_duplicates()
    num_team = {str(r.DriverNumber): str(r.Team) for r in pairs.itertuples(index=False)}

    grouped = {}
    for num, d in drivers.items():
        team = num_team.get(num)
        if team:
            grouped.setdefault(team, []).append(d)

    out = [{"team": team, "drivers": sorted(ds, key=lambda x: -x["count"]),
            "count": sum(x["count"] for x in ds)} for team, ds in grouped.items()]
    out.sort(key=lambda t: -t["count"])
    return out


def fetch_clip_rows(dataset_gp, racing_number, limit=40):
    """Rows for one driver at one GP, sorted by message time. `dataset_gp` must
    include the year exactly as stored, e.g. '2020 Turkish Grand Prix'. Reads the
    local parquet (audio bytes embedded) when available, else the HF /filter API."""
    import localset
    if localset.is_available():
        return localset.clip_rows(dataset_gp, racing_number, limit)
    gp = str(dataset_gp).replace("'", "''")
    num = str(racing_number).replace("'", "''")
    where = f"\"grand_prix\"='{gp}' AND \"racing_number\"='{num}'"
    rows, offset = [], 0
    while True:
        d = _filter(where, length=100, offset=offset)
        batch = [r["row"] for r in d.get("rows", [])]
        rows.extend(batch)
        total = d.get("num_rows_total", len(rows))
        offset += len(batch)
        if not batch or offset >= total or len(rows) >= limit:
            break
    rows.sort(key=lambda r: r["message_timestamp"])
    return rows[:limit]


def resolve_driver_number(year, gp_name, session, driver):
    """Accept a FastF1 abbreviation (HAM) or a car number (44) -> number string."""
    if str(driver).isdigit():
        return str(driver)
    import fastf1
    os.makedirs(".fastf1cache", exist_ok=True)
    fastf1.Cache.enable_cache(".fastf1cache")
    ses = fastf1.get_session(int(year), gp_name, session)
    ses.load(telemetry=False, weather=False, messages=False)
    nums = ses.laps.pick_drivers(str(driver))["DriverNumber"].unique().tolist()
    if not nums:
        raise ValueError(f"driver '{driver}' not found in {year} {gp_name}")
    return str(nums[0])


def abs_lap_starts(year, gp_name, session, racing_number):
    """{lap_number: absolute UTC start} for the driver. Needs telemetry loaded —
    that's what populates FastF1's absolute LapStartDate."""
    import fastf1
    import pandas as pd
    os.makedirs(".fastf1cache", exist_ok=True)
    fastf1.Cache.enable_cache(".fastf1cache")
    ses = fastf1.get_session(int(year), gp_name, session)
    ses.load(telemetry=True, weather=False, messages=False)
    laps = ses.laps.pick_drivers(str(racing_number))[["LapNumber", "LapStartDate"]].dropna()
    out = {}
    for _, lp in laps.iterrows():
        d = pd.Timestamp(lp["LapStartDate"])
        if d.tz is None:
            d = d.tz_localize("UTC")
        out[int(lp["LapNumber"])] = d
    return out


def _clip_to_lap(ts, lap_starts):
    """Lap whose start-window contains ts; clamps to the first lap for anything
    before the green flag (formation/grid radio)."""
    import pandas as pd
    t = pd.Timestamp(ts)
    if t.tz is None:
        t = t.tz_localize("UTC")
    laps = sorted(lap_starts)
    if not laps:
        return None
    chosen = laps[0]
    for n in laps:
        if lap_starts[n] <= t:
            chosen = n
        else:
            break
    return chosen


def import_driver_radio(year, gp_name, session, driver,
                        lap_from, lap_to, clips_dir="clips", limit=40, clear=True):
    """Fetch this driver's radio at this GP from the dataset, map each clip to its
    real lap, and download those inside [lap_from, lap_to] as lap<N>.mp3.

    `clear=True` wipes any existing clips first, so a fresh import never leaves
    behind clips from a previously selected driver/GP (which would otherwise get
    analysed against the wrong race). Returns {imported: [...], ...}; keeps one clip
    per lap (the earliest), since the pipeline maps one clip -> one lap."""
    from laps import local_clips

    number = resolve_driver_number(year, gp_name, session, driver)
    dataset_gp = f"{int(year)} {gp_name}"
    rows = fetch_clip_rows(dataset_gp, number, limit=limit)
    if not rows:
        return {"imported": [], "dataset_gp": dataset_gp, "number": number,
                "total_found": 0}

    lap_starts = abs_lap_starts(year, gp_name, session, number)

    os.makedirs(clips_dir, exist_ok=True)
    # only clear once we know the fetch succeeded, so a failed import keeps old clips
    if clear:
        for path in local_clips(clips_dir).values():
            try:
                os.remove(path)
            except OSError:
                pass

    # pick one clip per lap (earliest) inside the window. `audio[0]` is a signed
    # url over the API (`src`) or embedded bytes from the local parquet (`bytes`).
    picks = {}   # lap -> (src_or_bytes, when, transcription)
    for row in rows:
        lap = _clip_to_lap(row["message_timestamp"], lap_starts)
        if lap is None or not (lap_from <= lap <= lap_to) or lap in picks:
            continue
        audio = row["audio"][0]
        picks[lap] = (audio.get("src") or audio.get("bytes"),
                      row["message_timestamp"], row.get("transcription"))

    def _download(lap, src):
        dest = os.path.join(clips_dir, f"lap{lap}.mp3")
        if isinstance(src, (bytes, bytearray)):          # local parquet: write bytes
            with open(dest, "wb") as f:
                f.write(src)
        else:                                            # API: fetch the signed url
            urllib.request.urlretrieve(src, dest)
        return dest

    # download clips in parallel — the HF asset URLs are slow (~10s each), so doing
    # them sequentially made a full race take minutes; a thread pool cuts that hard.
    # (Local bytes write instantly; the pool is harmless there.)
    imported = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_download, lap, src): (lap, when, tr)
                   for lap, (src, when, tr) in picks.items()}
        for fut in concurrent.futures.as_completed(futures):
            lap, when, tr = futures[fut]
            try:
                dest = fut.result()
            except Exception:
                continue   # skip a clip that failed to download, keep the rest
            imported.append({"lap": lap, "file": os.path.basename(dest),
                             "when": when, "transcription": tr})

    imported.sort(key=lambda r: r["lap"])
    race_min = min(lap_starts) if lap_starts else lap_from
    race_max = max(lap_starts) if lap_starts else lap_to
    return {"imported": imported, "dataset_gp": dataset_gp, "number": number,
            "total_found": len(rows), "race_lap_min": race_min, "race_lap_max": race_max}
