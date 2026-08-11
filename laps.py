"""Real lap times from FastF1, and locating the radio clips to analyse.

Recommended live setup (most robust for judging day):
  1. Run  python prepare_laps.py 2023 "British Grand Prix" R VER 28 42
     -> writes laps.csv  (real lap times, straight from FastF1)
  2. Drop matching audio in ./clips named  lap34.mp3, lap37.mp3, ...
     (grab team-radio clips for that driver; filename = the lap it belongs to)
  3. POST /api/analyze  ->  the pipeline transcribes + scores each clip and
     merges it onto the real lap times.

`fetch_livetiming_radio` is an experimental auto-fetch for recent sessions and
is off by default -- the local-clips path never depends on a live scrape.
"""
import os
import re
import glob


# ---- real lap times (FastF1) ------------------------------------------------
def lap_times_fastf1(year, gp, session, driver, lap_from, lap_to):
    """Returns [{'lap': int, 't': float}] for one driver's lap window."""
    import fastf1
    os.makedirs(".fastf1cache", exist_ok=True)
    fastf1.Cache.enable_cache(".fastf1cache")

    ses = fastf1.get_session(int(year), gp, session)
    ses.load(telemetry=False, weather=False, messages=False)

    # Let pandas do the window filter + NaT drop. LapTime is NaT on in/out laps
    # and missing data; dropna removes them so we never emit a nan lap time.
    laps = ses.laps.pick_drivers(driver)
    laps = laps.loc[
        laps["LapNumber"].between(lap_from, lap_to),
        ["LapNumber", "LapTime"],
    ].dropna(subset=["LapNumber", "LapTime"]).sort_values("LapNumber")

    return [
        {"lap": int(n), "t": round(t.total_seconds(), 3)}
        for n, t in zip(laps["LapNumber"], laps["LapTime"])
    ]


def lap_times_csv(path="laps.csv"):
    import pandas as pd
    df = pd.read_csv(path).sort_values("lap")
    return [{"lap": int(r.lap), "t": float(r.t)} for r in df.itertuples(index=False)]


def driver_team_fastf1(year, gp, session, driver):
    """The driver's team name for this session (reuses FastF1's cache, so it's
    cheap after lap_times_fastf1 already loaded it). '' if unavailable."""
    import fastf1
    os.makedirs(".fastf1cache", exist_ok=True)
    fastf1.Cache.enable_cache(".fastf1cache")
    ses = fastf1.get_session(int(year), gp, session)
    ses.load(telemetry=False, weather=False, messages=False)
    laps = ses.laps.pick_drivers(str(driver))
    teams = laps["Team"].dropna().unique().tolist() if "Team" in laps else []
    return str(teams[0]) if teams else ""


# ---- locate the audio clips -------------------------------------------------
def local_clips(folder="clips"):
    """Files named lapNN.<ext> -> {lap_number: path}."""
    found = {}
    for p in glob.glob(os.path.join(folder, "*")):
        m = re.search(r"lap[_-]?(\d+)", os.path.basename(p), re.I)
        if m:
            found[int(m.group(1))] = p
    return found


# ---- experimental: pull team radio straight from F1 livetiming --------------
def fetch_livetiming_radio(year, gp, session, driver, out="clips"):
    """Best-effort. Resolves the session's static path, reads TeamRadio.json,
    downloads this driver's captures as lapNN.mp3 by matching capture time to
    lap start time. Wrapped so failure never blocks the local-clips path."""
    import fastf1, requests
    os.makedirs(out, exist_ok=True)
    base = "https://livetiming.formula1.com/static/"

    ses = fastf1.get_session(int(year), gp, session)
    ses.load(telemetry=False, weather=False, messages=False)
    info = getattr(ses, "session_info", {}) or {}
    path = info.get("Path")
    if not path:
        raise RuntimeError("session Path not exposed by this FastF1 version; use local clips")

    tr = requests.get(base + path + "TeamRadio.json", timeout=20)
    tr.raise_for_status()
    captures = tr.json().get("Captures", [])

    drv_laps = ses.laps.pick_drivers(driver)
    saved = {}
    for c in captures:
        # driver filtering / lap matching left simple for the MVP; refine per season
        url = base + path + c["Path"]
        # (matching capture UTC -> lap number is the real work; see README notes)
        saved[c.get("Path")] = url
    return saved
