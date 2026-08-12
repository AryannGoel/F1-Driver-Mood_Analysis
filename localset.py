"""Local mirror of the HF team-radio dataset, read straight from the downloaded
parquet shards instead of the (slow, 500-prone) datasets-server `/filter` API.

Run `python _download_dataset.py` once to pull the ~2.57 GB of parquet into
`.hfcache/f1-team-radio/`; after that `is_available()` is True and `dataset.py`
delegates catalog/driver/clip lookups here — dropdowns become instant and imports
read the audio bytes embedded in the parquet, so no per-clip network download.

Row shapes intentionally match the API rows `dataset.py` already consumes:
  gp_rows  -> [{racing_number, driver_id, message_timestamp}]
  clip_rows-> [{message_timestamp, transcription, audio:[{bytes, path}]}]  (sorted)
"""
import os
import glob
import functools

LOCAL_DIR = os.environ.get(
    "LOCAL_DATASET_DIR", os.path.join(os.path.dirname(__file__), ".hfcache", "f1-team-radio")
)

# metadata columns only — small, safe to hold all rows in memory (the big `audio`
# column is read on demand, per driver, so we never load 2.5 GB into RAM).
_META_COLS = ["id", "driver_id", "racing_number", "grand_prix", "message_timestamp", "transcription"]


def _parquet_files():
    """The dataset shards only — never the slim `_meta.parquet` cache, which lacks
    the audio column and would poison the unified schema pyarrow infers."""
    files = glob.glob(os.path.join(LOCAL_DIR, "**", "*.parquet"), recursive=True)
    return sorted(f for f in files if os.path.basename(f) != "_meta.parquet")


def is_available() -> bool:
    """True once the parquet shards have been downloaded locally."""
    return bool(_parquet_files())


@functools.lru_cache(maxsize=1)
def _meta():
    """Every row's metadata (no audio) as one pandas DataFrame, loaded + cached once.

    Scanning the metadata columns out of the 2.4 GB of shards takes ~18 s, so the
    slim result is persisted to `_meta.parquet` (a few hundred KB) — every later
    server start then loads it in milliseconds instead of rescanning."""
    import pyarrow.parquet as pq
    import pandas as pd
    cache = os.path.join(LOCAL_DIR, "_meta.parquet")
    if os.path.exists(cache):
        try:
            return pd.read_parquet(cache)
        except Exception:
            pass                                   # corrupt cache -> rebuild below
    frames = [pq.read_table(f, columns=_META_COLS).to_pandas() for f in _parquet_files()]
    df = pd.concat(frames, ignore_index=True)
    df["grand_prix"] = df["grand_prix"].astype(str)
    df["racing_number"] = df["racing_number"].astype(str)
    try:
        df.to_parquet(cache, index=False)
    except Exception:
        pass                                       # cache is an optimisation, not required
    return df


def catalog():
    """Years + Grand Prix from the local grand_prix value counts (same shape as the
    API's catalog): {years:[...newest first], by_year:{year:[{name, dataset_gp, count}]}}."""
    freq = _meta()["grand_prix"].value_counts()
    by_year = {}
    for dataset_gp, count in freq.items():
        parts = str(dataset_gp).split(" ", 1)              # "2020 Turkish Grand Prix"
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        year, name = parts
        by_year.setdefault(year, []).append(
            {"name": name, "dataset_gp": dataset_gp, "count": int(count)})
    for year in by_year:
        by_year[year].sort(key=lambda g: g["name"])
    return {"years": sorted(by_year, reverse=True), "by_year": by_year}


def gp_rows(dataset_gp):
    """All rows for one GP (racing_number, driver_id, message_timestamp)."""
    m = _meta()
    sub = m[m["grand_prix"] == str(dataset_gp)]
    return [{"racing_number": r.racing_number, "driver_id": r.driver_id,
             "message_timestamp": r.message_timestamp}
            for r in sub.itertuples(index=False)]


def clip_rows(dataset_gp, racing_number, limit=40):
    """One driver's rows at one GP, sorted by message time, WITH the audio bytes
    (read on demand from the parquet). Shaped like the API rows so import_driver_radio
    can treat `row['audio'][0]` uniformly (bytes here, a signed src url over the API)."""
    import pyarrow.dataset as pds
    import pyarrow.compute as pc
    dataset = pds.dataset(_parquet_files(), format="parquet")
    flt = (pc.field("grand_prix") == str(dataset_gp)) & \
          (pc.field("racing_number") == str(racing_number))
    tbl = dataset.to_table(filter=flt, columns=["message_timestamp", "transcription", "audio"])
    df = tbl.to_pandas().sort_values("message_timestamp").head(limit)

    rows = []
    for r in df.itertuples(index=False):
        a = r.audio or {}
        rows.append({
            "message_timestamp": r.message_timestamp,
            "transcription": r.transcription,
            "audio": [{"bytes": a.get("bytes"), "path": a.get("path")}],
        })
    return rows
