<!-- Hugging Face Space config (ignored on GitHub; read by HF Spaces). -->
---
title: Pit Wall — The Silent Co-Driver
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Pit Wall — The Silent Co-Driver

> Reads a Formula 1 **driver's** stress from real team radio and lines it up against their lap times — so you can *see* the moment the pressure hit and what it cost on track.

Built for the **Geek Room · AI Race Month** hackathon (Problem Statement 01 — *The Silent Co-Driver*).

---

## 1. What it is

Every F1 driver is talking to their race engineer over the radio all race long. Buried in that audio is a signal nobody watches in real time: **how the driver actually sounds** — calm, focused, stressed, spent. Pit Wall pulls that signal out and puts it next to the lap-time trace, so a stress spike on the radio and the slower lap that follows sit on the same chart.

The core insight the dashboard is built around: **stress on the radio → the next lap is usually slower.** The tyre never recovers.

It is a small thing that works end to end:

- **Real audio** — genuine team radio from a Hugging Face dataset.
- **Real lap times** — pulled from FastF1 (official timing).
- **Real AI** — two Hugging Face models do the perception (speech → text, speech → emotion).
- **One screen** — pick a driver + Grand Prix, import their radio, hit *Run*, and read the stress-vs-laptime story.

---

## 2. How it works (end to end)

```
                    ┌─────────────────────────── Pit Wall ───────────────────────────┐
   You pick:        │                                                                 │
   Year ▸ GP ▸      │   1. IMPORT ─ pull this driver's radio for this GP from the      │
   Driver           │      Hugging Face dataset, and map each clip to the lap it was   │
        │           │      actually sent on (clip UTC time vs FastF1 lap start time).  │
        ▼           │                                                                 │
   [IMPORT REAL     │   2. For each clip, on RUN ANALYSIS:                             │
    RADIO]          │        a. isolate the DRIVER's in-car voice from the engineer    │
        │           │           (energy + noisiness), so the calm pit wall doesn't     │
        ▼           │           skew the reading                                       │
   [RUN ANALYSIS]   │        b. Whisper  →  transcript                                 │
        │           │        c. wav2vec2 →  emotion  →  mood.py  →  stress 0–100        │
        ▼           │   3. Merge onto real FastF1 lap times → one JSON contract        │
   Dashboard        │   4. UI plots stress vs lap time, plays the clip, shows the tone │
                    └─────────────────────────────────────────────────────────────────┘
```

**The pipeline per clip** (`pipeline.py`):

1. **Driver-voice isolation** (`driver_audio`) — team radio carries two voices: the **driver** (in-car mic — loud, noisy from engine/wind, heavily compressed) and the **race engineer** (clean pit-wall feed). We load the clip at 16 kHz, split it into voiced utterances, and *only when the clip clearly splits into a clean vs noisy pair of sources* keep the noisier (in-car) speech. A single-speaker clip is left whole, so this never hurts the common case.
2. **ASR** — `openai/whisper-base.en` transcribes the driver audio → text.
3. **SER (tone of voice)** — `superb/wav2vec2-base-superb-er` classifies the driver audio → emotion probabilities.
4. **Text emotion (the words)** — `j-hartmann/emotion-english-distilroberta-base` classifies the transcript → emotion probabilities.
5. **Domain layer** (`mood.py`) — **fuses** the voice + text emotions, then maps them to race-relevant **mood** (`calm / focused / stressed / tired`) + a **0–100 stress score**. Later laps skew a distress reading toward *fatigue* rather than acute stress.

The result for each lap: `{ lap, t (sec), stress, mood, radio (transcript), tone (raw emotion), confidence }`.

---

## 3. The dataset

**[`MikCil/f1-team-radio`](https://huggingface.co/datasets/MikCil/f1-team-radio)** — 14,681 real F1 team-radio clips, 2018–2025, 149 Grands Prix, 43 drivers.

| Field | Meaning |
|-------|---------|
| `audio` | the radio clip (mp3, 16 kHz) |
| `transcription` | ground-truth transcript |
| `driver_id` | e.g. `LEWHAM01` (chars 3–6 = the standard code, `HAM`) |
| `racing_number` | car number, e.g. `44` |
| `grand_prix` | includes the year, e.g. `2020 Turkish Grand Prix` |
| `message_timestamp` | absolute UTC time the message was sent |
| `session_date`, `race_id` | session metadata |

How we use it (`dataset.py`):

- **Catalog** — the Year ▸ Grand Prix ▸ Driver dropdowns are built from the dataset itself (via the HF *datasets-server* `/statistics` and `/filter` APIs), so you can **only pick combinations that actually have audio**. The dataset's GP names match FastF1's event schedule 100%, so every listed GP is also loadable.
- **Import** — for a chosen driver+GP we pull just that handful of rows (not the whole 2.5 GB), download the audio in **parallel** (the HF asset URLs are slow one-by-one), and map each clip to a lap.
- **Lap alignment** — each clip's `message_timestamp` is matched against FastF1's **absolute per-lap start time** (`LapStartDate`, which requires telemetry to be loaded), so a clip is filed under the exact lap it happened on — not guessed.
- **Optional local copy (much faster, offline)** — the HF datasets-server API is slow and 500s intermittently, so `python _download_dataset.py` pulls the 5 parquet shards (~2.57 GB, audio embedded) into `.hfcache/`. When present, `localset.py` serves the catalog/driver/clip lookups straight from the parquet: **dropdowns drop from ~15–160 s (often failing) to ~0.2 s, and import from minutes to ~5 s.** `dataset.py` falls back to the API automatically when the parquet isn't downloaded, so nothing else changes. Analysis speed is unaffected — that's CPU model inference, not data fetching.

---

## 4. The AI/ML — all on Hugging Face

The hackathon requires most of the AI/ML to run on Hugging Face. Here it is **100%** of the perception:

| Stage | Model | Library |
|-------|-------|---------|
| Speech → text | `openai/whisper-base.en` | `transformers` pipeline |
| Speech → emotion (tone of voice) | `superb/wav2vec2-base-superb-er` | `transformers` pipeline |
| Text → emotion (the words) | `j-hartmann/emotion-english-distilroberta-base` | `transformers` pipeline |
| Data | `MikCil/f1-team-radio` | HF datasets-server (or local parquet) |

**Three HF models, two modalities.** Team radio is short and compressed, so the voice model is often unsure — so we also read the emotion in the **transcript** with a text-classification model and **fuse** the two (`mood.fuse`, default 60% voice / 40% words). A clip where the voice sounds flat but the words are *"Don't [...] me, man"* now reads as elevated, not calm — the words steady the read.

- `openai/whisper-base.en` is a **Hugging Face Hub repo id** (an org namespace), **not** the OpenAI API — weights download from huggingface.co and run locally on CPU. No API key, no external calls.
- Swap any model with the `ASR_MODEL` / `SER_MODEL` / `TXT_EMO_MODEL` env vars.
- The **only** non-HF logic is `mood.py` — a hand-written rule layer that fuses the two emotion signals and turns them into *race* stress/fatigue. That's the intended domain glue (the "not a raw tool call" part), not a competing ML model.

> **Why superb and not the popular `ehcalabres/...` emotion model?** That checkpoint no longer loads its classifier head under `transformers` 5.x (the head params are missing → randomly initialised → meaningless near-uniform output). `superb/wav2vec2-base-superb-er` is an official SUPERB model whose trained head loads cleanly and gives confident, discriminating emotions, and its 4-class IEMOCAP vocabulary is exactly what `mood.py` maps.

---

## 5. Setup & run

```bash
pip install -r requirements.txt
python -m uvicorn app:app --port 8000
```

Then open **http://localhost:8000**.

**Optional but recommended — download the dataset locally:**

```bash
python _download_dataset.py    # ~2.57 GB, one time → .hfcache/
```

This makes the dropdowns and imports near-instant and immune to the Hugging Face datasets-server being slow or down. Skip it and everything still works via the API, just slower. (On Windows, `run.bat` does the pip install + launch for you.)

- First run downloads the three HF models (~1.5 GB, cached afterwards).
- CPU-only is fine and intended for reliability; a full analysis of a lap window takes tens of seconds on CPU (~6 s per clip).
- Audio is decoded with `soundfile` + `soxr` (no `ffmpeg` and no `numba` needed).
- FastF1 caches session data under `.fastf1cache/`; imported clips land in `clips/`; the optional local dataset lives in `.hfcache/`.

**Using the dashboard:**

1. Choose **One Driver** or **Whole Team**, then pick **Year → Grand Prix → Session → Driver/Team** (every dropdown only shows options that have real audio).
2. Click **⇩ Import Real Radio** — pulls the clips and maps them to laps (team mode imports both drivers into separate folders). The lap dropdowns fill with the race's laps.
3. (Optional) Manually **↑ Add Clip** or **● Record** a clip for a lap, or **✕ Remove** / **✕ Clear All Clips** to reset.
4. Set the **From/To lap** window and click **▶ Run Analysis**.
5. Read the dashboard: stress-vs-laptime chart, per-lap transcript, driver **tone + confidence**, and **▶ play the actual clip**.
> ⚠️ **Note:** Loading the timings for the selected lap session can take a few minutes, so please be patient while it fetches the data.
---

## 6. Project structure

```
app.py            FastAPI app + all endpoints (UI, health, clips, import, analyze, dataset catalog)
pipeline.py       HF inference — Whisper (ASR) + wav2vec2 (SER) + driver-voice isolation
mood.py           domain layer: emotion probabilities → mood + 0–100 stress
dataset.py        MikCil/f1-team-radio: catalog, driver lookup, import + lap alignment (API, local fallback)
localset.py       optional local mirror — reads catalog/clips/audio from the downloaded parquet shards
_download_dataset.py  one-time: download the dataset parquet locally (~2.57 GB → .hfcache/)
laps.py           FastF1 real lap times, absolute lap-start times, driver→team, local clip discovery
schemas.py        pydantic output contract (Meta + Lap)
prepare_laps.py   CLI to export a driver's laps to laps.csv
frontend/index.html   the served single-page UI (no build step)
requirements.txt  dependencies
clips/            imported/uploaded audio (lapNN.mp3)
.hfcache/         optional downloaded dataset (gitignored)
```

### API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | the dashboard UI |
| `GET` | `/api/health` | liveness |
| `GET` | `/api/dataset/catalog` | years + Grands Prix (that have audio) for the dropdowns |
| `GET` | `/api/dataset/drivers?gp=…` | drivers with radio at a GP, richest first |
| `GET` | `/api/dataset/sessions?year=&gp=` | sessions of a GP that actually have radio |
| `GET` | `/api/dataset/teams?year=&gp=&session=` | constructors + their drivers (team mode) |
| `GET` | `/api/dataset/lap-range?year=&gp=&session=` | real lap range for a session (fills the FROM/TO dropdowns) |
| `POST` | `/api/import` | pull a driver's radio for a GP, align to laps, download to `clips/` |
| `POST` | `/api/analyze` | run the pipeline over the lap window → the dashboard JSON |
| `GET` | `/api/clips` | list current clips |
| `POST` | `/api/clips` | upload/record a clip for a lap |
| `GET` | `/api/clips/{lap}/audio` | stream a lap's clip (for in-browser playback) |
| `DELETE` | `/api/clips/{lap}` | remove one lap's clip |
| `DELETE` | `/api/clips` | **clear all clips (reset)** |

---

## 7. Design decisions & honest limits

Say these out loud in the pitch — they're features, not bugs:

- **Speech-emotion from acted-emotion models is noisy.** Treat the tone as a **signal, not a verdict**. Most F1 radio reads neutral/calm because drivers are composed on the radio; the clear moments (frustration, fatigue, elation) are where it shines.
- **Driver isolation is a heuristic, not diarization.** It uses audio energy + spectral flatness (in-car mic vs clean pit-wall feed) and only trims when there's a clear two-speaker split. It can't isolate a driver who never speaks in a clip. The accurate alternative (gated `pyannote` diarization) was deliberately not used — it needs an HF token + accepting its terms.
- **Lap alignment is real** but depends on FastF1 exposing absolute lap times (telemetry load), and on the dataset's GP name matching FastF1's — which it does for the seasons checked.
- **CPU by design** — reliable over a demo network beats a flaky GPU download.
- **No numba in the audio path** — clips are decoded/resampled with `soundfile` + `soxr` and framed in NumPy, deliberately avoiding `librosa`, whose `numba` dependency ships a native DLL that Windows Application Control blocks on some locked-down machines (it used to crash every analysis). `librosa` stays installed only because `transformers` imports it lazily.

---

## 8. Concept evolution (kept up to date as the project changes)

This section tracks the **major shifts in what the project is**, newest first — so the README stays honest as concepts change.

- **Local dataset option** — `python _download_dataset.py` caches the dataset's parquet shards locally; `localset.py` then serves catalog/driver/clip lookups + audio from disk (with automatic API fallback), cutting dropdowns from ~15–160 s to ~0.2 s and imports from minutes to ~5 s, and eliminating the datasets-server 500s.
- **Numba-free audio** — replaced `librosa` audio loading + feature extraction with `soundfile` + `soxr` + NumPy, so analysis runs on locked-down Windows where numba's native DLL is blocked (previously every analysis 500'd at the first clip).
- **Meaningful stats for composed drivers** — the per-driver stats panel now falls back to a driver's PEAK STRESS / PEAK RADIO moment when they never cross the "stressed" threshold, instead of blank dashes (so e.g. a calm-all-race driver still reports real numbers).
- **Clearer failures** — dropdown/import errors now surface the real reason (and retry the transient HF 500s) instead of a cryptic JSON-parse error or a silent ✕.
- **Fresh start, real lap ranges** — the UI now loads with nothing pre-selected (no default year/GP/driver/lap); you pick a year → GP → and the session's real lap range drives the FROM/TO dropdowns from FastF1 (`/api/dataset/lap-range`), instead of hardcoded lap numbers.
- **Sessions with audio only** — the Session dropdown is now built per-GP: each clip's UTC timestamp is bucketed against FastF1's session schedule, so only sessions that actually have radio are offered (e.g. 2023 British → Qualifying + Race, no empty practice sessions).
- **Team mode (two drivers at once)** — a Driver / Team toggle: pick a constructor and both its drivers are imported (into separate `clips/<DRIVER>` folders) and analysed into their own full dashboards, stacked for side-by-side reading. The dashboard is now a reusable component.
- **Multimodal emotion (more Hugging Face)** — added a third HF model, `j-hartmann/emotion-english-distilroberta-base`, reading emotion from the *transcript*, fused with the voice-tone model. The mood now reflects both how the driver *sounds* and what they *say* — so flat-voiced but frustrated calls (e.g. "brakes aren't working, man") no longer read as calm.
- **Reset control** — added *Clear All Clips* to wipe the retrieved clip set in one click (plus per-lap remove).
- **Clearer controls** — Session shows readable labels (Race / Qualifying / …); the three lap fields are dropdowns of real laps; the tone display shows the driver's *actual* detected emotion + model confidence (replacing a faked confidence number).
- **Analyse the driver, not the team** — the pipeline now isolates the driver's in-car voice before ASR + SER, so the engineer's calm delivery no longer skews the reading (e.g. lap 58 of 2020 Turkey flips from the engineer's "Get in there Lewis!" to Hamilton's own tired, emotional words).
- **Real dataset integration** — wired in `MikCil/f1-team-radio`: cascading Year→GP→Driver dropdowns, timestamp-to-lap alignment via FastF1, parallel downloads, and real team names from FastF1.
- **Emotion model fix** — switched the SER default from the broken `ehcalabres/...` checkpoint (uninitialised head → noise) to `superb/wav2vec2-base-superb-er` (trained head, discriminating output).
- **Live-only** — removed the offline demo endpoint and fallback; the app is now the real pipeline end to end.
- **Origins** — started as a two-mode MVP (offline demo + live pipeline) sharing one JSON contract and one UI.

---

*Pit Wall reads the driver, not the noise.*
