"""Pit Wall API — The Silent Co-Driver.

  GET  /                     -> the Pit Wall UI (served same-origin)
  GET  /api/health
  GET  /api/clips            -> list uploaded voice clips
  POST /api/clips            -> upload/record a voice clip for a lap
  POST /api/analyze          -> real FastF1 laps + Whisper + wav2vec2

Run:  uvicorn app:app --reload --port 8000   then open http://localhost:8000
"""
import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from schemas import Stint

app = FastAPI(title="Pit Wall API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

FRONTEND = os.path.join(os.path.dirname(__file__), "frontend", "index.html")


@app.on_event("startup")
def _prewarm_dataset():
    """Warm the dataset catalog + the default GP's drivers in the background so the
    UI dropdowns are ready fast (the HF /filter scan is slow the first time)."""
    import threading

    def warm():
        try:
            from dataset import catalog, drivers_for_gp
            catalog()
            drivers_for_gp("2020 Turkish Grand Prix")
        except Exception:
            pass

    threading.Thread(target=warm, daemon=True).start()


@app.get("/")
def index():
    """Serve the Pit Wall UI from the same origin as the API."""
    # no-store so the browser never serves a stale UI after an update
    return FileResponse(FRONTEND, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return {"ok": True}


# ---- voice-clip management (used by the LIVE CONTROL panel in the UI) --------
ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".oga", ".webm", ".flac", ".aac"}


def _safe_clips_dir(clips_dir: str) -> str:
    """Keep uploads inside the project — reject absolute/parent paths."""
    base = os.path.dirname(__file__)
    full = os.path.normpath(os.path.join(base, clips_dir))
    if not full.startswith(base):
        raise HTTPException(400, "invalid clips_dir")
    return full


@app.get("/api/clips")
def list_clips(clips_dir: str = "clips"):
    """Current voice clips on disk, as [{lap, file}] sorted by lap."""
    from laps import local_clips
    found = local_clips(_safe_clips_dir(clips_dir))
    return {"clips": [{"lap": lap, "file": os.path.basename(p)}
                      for lap, p in sorted(found.items())]}


@app.post("/api/clips")
async def upload_clip(lap: int = Form(...), file: UploadFile = File(...),
                      clips_dir: str = Form("clips")):
    """Save an uploaded/recorded radio clip as lap<N>.<ext>. Replaces any
    existing clip for the same lap so one lap maps to exactly one file."""
    ext = os.path.splitext(file.filename or "")[1].lower() or ".webm"
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(400, f"unsupported audio type '{ext}' (allowed: {sorted(ALLOWED_AUDIO)})")

    folder = _safe_clips_dir(clips_dir)
    os.makedirs(folder, exist_ok=True)

    # drop any prior clip for this lap (regardless of its extension)
    from laps import local_clips
    for existing_lap, path in local_clips(folder).items():
        if existing_lap == lap:
            try:
                os.remove(path)
            except OSError:
                pass

    dest = os.path.join(folder, f"lap{lap}{ext}")
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"ok": True, "lap": lap, "file": os.path.basename(dest)}


@app.get("/api/clips/{lap}/audio")
def clip_audio(lap: int, clips_dir: str = "clips"):
    """Serve the raw audio for one lap's clip so the UI can actually play it."""
    from laps import local_clips
    path = local_clips(_safe_clips_dir(clips_dir)).get(lap)
    if not path or not os.path.exists(path):
        raise HTTPException(404, f"no clip for lap {lap}")
    return FileResponse(path)


@app.delete("/api/clips")
def clear_clips(clips_dir: str = "clips"):
    """Remove ALL retrieved/uploaded clips (reset)."""
    from laps import local_clips
    removed = []
    for path in local_clips(_safe_clips_dir(clips_dir)).values():
        try:
            os.remove(path)
            removed.append(os.path.basename(path))
        except OSError:
            pass
    return {"ok": True, "removed": removed}


@app.delete("/api/clips/{lap}")
def delete_clip(lap: int, clips_dir: str = "clips"):
    """Remove the clip(s) for one lap."""
    from laps import local_clips
    removed = []
    for existing_lap, path in local_clips(_safe_clips_dir(clips_dir)).items():
        if existing_lap == lap:
            try:
                os.remove(path)
                removed.append(os.path.basename(path))
            except OSError:
                pass
    return {"ok": True, "removed": removed}


@app.get("/api/dataset/catalog")
def dataset_catalog():
    """Years + Grand Prix available in the HF dataset, for the UI dropdowns."""
    from dataset import catalog
    try:
        return catalog()
    except Exception as e:
        raise HTTPException(502, f"dataset catalog failed: {e}")


@app.get("/api/dataset/drivers")
def dataset_drivers(gp: str):
    """Drivers with radio at one GP (gp = dataset name incl. year), richest first."""
    from dataset import drivers_for_gp
    try:
        return {"gp": gp, "drivers": drivers_for_gp(gp)}
    except Exception as e:
        raise HTTPException(502, f"dataset drivers failed: {e}")


@app.get("/api/dataset/sessions")
def dataset_sessions(year: int, gp: str):
    """Sessions of this GP that actually have radio (so the UI hides empty ones)."""
    from dataset import sessions_for_gp
    try:
        return {"sessions": sessions_for_gp(year, gp, f"{year} {gp}")}
    except Exception as e:
        raise HTTPException(502, f"dataset sessions failed: {e}")


@app.get("/api/dataset/teams")
def dataset_teams(year: int, gp: str, session: str = "R"):
    """Constructors at one GP, each with its drivers (for team-mode analysis)."""
    from dataset import teams_for_gp
    try:
        return {"teams": teams_for_gp(year, gp, session, f"{year} {gp}")}
    except Exception as e:
        raise HTTPException(502, f"dataset teams failed: {e}")


class ImportReq(BaseModel):
    """Pull real team radio for one driver+GP from the HF dataset MikCil/f1-team-radio."""
    year: int = 2020
    gp: str = "Turkish Grand Prix"          # FastF1 event name; year is added for the dataset
    session: str = "R"
    driver: str = "HAM"                      # abbreviation or car number
    lap_from: int = 1
    lap_to: int = 58
    clips_dir: str = "clips"
    limit: int = 40


@app.post("/api/import")
def import_radio(req: ImportReq):
    """Fetch this driver's radio at this GP from the Hugging Face dataset, map each
    clip to the lap it was sent on (FastF1 absolute lap times), and save the ones
    in [lap_from, lap_to] into clips_dir. RUN ANALYSIS then works on real audio."""
    from dataset import import_driver_radio
    try:
        result = import_driver_radio(
            req.year, req.gp, req.session, req.driver,
            req.lap_from, req.lap_to, _safe_clips_dir(req.clips_dir), req.limit,
        )
    except Exception as e:
        raise HTTPException(502, f"dataset import failed: {e}")

    if not result["imported"]:
        raise HTTPException(404,
            f"no radio clips mapped into laps {req.lap_from}-{req.lap_to} "
            f"for '{req.driver}' at {result.get('dataset_gp')} "
            f"(found {result.get('total_found', 0)} clips for that driver)")
    return result


class AnalyzeReq(BaseModel):
    year: int = 2020
    gp: str = "Turkish Grand Prix"
    session: str = "R"
    driver: str = "HAM"
    lap_from: int = 1
    lap_to: int = 58
    clips_dir: str = "clips"
    team: str = "MERCEDES"


@app.post("/api/analyze", response_model=Stint)
def analyze(req: AnalyzeReq):
    """Live pipeline. Needs transformers+torch+fastf1 installed and audio in clips_dir."""
    from laps import lap_times_fastf1, local_clips, driver_team_fastf1
    from pipeline import analyse_clip

    try:
        lap_rows = lap_times_fastf1(req.year, req.gp, req.session,
                                    req.driver, req.lap_from, req.lap_to)
    except Exception as e:
        raise HTTPException(502, f"FastF1 lap load failed: {e}")

    if not lap_rows:
        raise HTTPException(404, "no laps in that window for that driver")

    clips = local_clips(req.clips_dir)
    if not clips:
        raise HTTPException(400, f"no radio clips found in '{req.clips_dir}' (name them lap34.mp3 ...)")

    lap_min, lap_max = lap_rows[0]["lap"], lap_rows[-1]["lap"]
    span = max(1, lap_max - lap_min)

    # real team from FastF1 (session already cached above), so it isn't hardcoded
    try:
        team = driver_team_fastf1(req.year, req.gp, req.session, req.driver) or req.team
    except Exception:
        team = req.team

    laps = []
    for r in lap_rows:
        n = r["lap"]
        if n in clips:
            progress = (n - lap_min) / span
            row = analyse_clip(clips[n], n, r["t"], progress)
        else:
            row = {"lap": n, "t": round(r["t"], 1), "stress": 0, "mood": "calm", "radio": None}
        laps.append(row)

    # laps with no radio inherit a neutral-calm baseline; carry the last mood softly
    return {
        "meta": {
            "driver": req.driver, "team": team,
            "session": f"{req.gp} · {req.session}", "compound": "HARD",
            "stint": f"L{lap_min}-L{lap_max}", "source": "live",
        },
        "laps": laps,
    }


if __name__ == "__main__":
    # `python app.py` — honours $PORT so it runs as-is on Spaces/Render/Railway/etc.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
