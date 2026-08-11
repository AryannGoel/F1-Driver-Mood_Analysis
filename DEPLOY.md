# Deploying Pit Wall

## The honest constraint

Pit Wall is **not a static site** — the backend loads ~1 GB of Hugging Face models
into memory and runs ~60 s of CPU inference per analysis, and it downloads FastF1
timing data at runtime. That rules out **Vercel / Netlify / GitHub Pages**, which are
built for static files or short-lived serverless functions (size, RAM, and timeout
limits are all exceeded). Those hosts can serve the *frontend*, but not this backend.

**Recommended free host: Hugging Face Spaces (Docker).** It's free, gives ~16 GB RAM
and 50 GB disk (enough for the models), runs the whole app + UI from one origin, and
the project is already Hugging Face–based. A `Dockerfile` is included and works on any
container host.

---

## Option A — Hugging Face Spaces (recommended, free)

1. Create a new Space → **SDK: Docker** → **Blank**.
2. Push this repo to the Space (or connect the GitHub repo). Add this YAML to the
   **top of the Space's `README.md`** (Spaces reads it for config):
   ```yaml
   ---
   title: Pit Wall
   emoji: 🏎️
   colorFrom: red
   colorTo: gray
   sdk: docker
   app_port: 7860
   ---
   ```
3. The Space builds the `Dockerfile` and starts the app on port 7860.
4. First analysis is slow (models download once, ~1 GB, then cached). CPU inference
   is ~60 s per lap window — fine for a demo. Note the Space filesystem is ephemeral,
   so imported clips reset on rebuild (just re-import).

## Option B — any container host (Render / Railway / Fly.io)

The `Dockerfile` runs anywhere. The app honours `$PORT`.
- **Render / Railway / Fly:** New Web Service → Docker → this repo. Set the start
  command to `uvicorn app:app --host 0.0.0.0 --port $PORT` (or use the Docker `CMD`).
- ⚠️ **RAM:** torch + three models need **≳ 2 GB RAM**. Render's *free* tier (512 MB)
  is too small — use a paid instance there, or prefer HF Spaces' free 16 GB.

## Option C — split: frontend on Vercel + backend on a Space

Only if you specifically want Vercel. Host the backend on a Space (Option A), then
serve `frontend/index.html` on Vercel and point its API calls at the Space URL
(the API already sends permissive CORS headers). One origin (Option A) is simpler.

---

## Local run

```bash
pip install -r requirements.txt
python -m uvicorn app:app --port 8000     # or: python app.py
# open http://localhost:8000
```

## Not committed (see `.gitignore`)
- `clips/` audio — copyrighted F1 team radio, fetched at runtime; not redistributed.
- `.fastf1cache/`, `.hfcache/` — large caches rebuilt on demand.
