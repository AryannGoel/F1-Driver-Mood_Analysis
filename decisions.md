# Decisions — Pit Wall (The Silent Co-Driver)

A running log of the **engineering decisions** behind the code, with the reasoning,
the alternatives that were rejected, and the trade-offs accepted. Newest themes first.
Kept alongside [README.md](README.md) (§8 tracks *what the project is*; this tracks *why the code is the way it is*).

Format per entry: **Decision → Why → Alternatives / Trade-off.**

---

## 1. Emotion & analysis pipeline

### 1.1 Swap the speech-emotion model (`ehcalabres` → `superb/wav2vec2-base-superb-er`)
- **Decision:** default SER model is `superb/wav2vec2-base-superb-er`.
- **Why:** the popular `ehcalabres/wav2vec2-lg-xlsr-...-emotion` checkpoint no longer loads its classifier head under `transformers` 5.x — the head params are missing, so they are randomly initialised and the output is a meaningless near-uniform distribution (verified: two opposite clips gave identical scores). "wav2vec2 → tone" was effectively fake. `superb` is an official SUPERB model whose trained head loads cleanly and gives confident, discriminating emotions; its 4-class IEMOCAP vocabulary (angry/happy/neutral/sad) is already what `mood.py` maps.
- **Alternatives / trade-off:** pin an old `transformers` (rejected — would break other things); keep `ehcalabres` (rejected — broken). Old model still selectable via `SER_MODEL`.

### 1.2 Multimodal emotion — fuse voice tone + transcript emotion
- **Decision:** add a third HF model, `j-hartmann/emotion-english-distilroberta-base`, on the transcript, and fuse it with the voice model in `mood.fuse` (default 60% voice / 40% words).
- **Why:** team radio is short and compressed, so the acoustic model is often unsure; the words carry clear emotion ("brakes aren't working, man" is unmistakably frustrated). Fusing lifts flat-voiced-but-angry calls out of "calm". More HF, more accurate — both goals.
- **Alternatives / trade-off:** voice-only (rejected — misses linguistic signal). Adds one small model load; no new dependency (rides on `transformers`).

### 1.3 Analyse the driver, not the team — isolate the in-car voice
- **Decision:** before ASR + SER, isolate the driver's (in-car) speech from the race engineer's (pit-wall) speech using an **energy + spectral-flatness heuristic**, and only trim when a clip clearly splits into two sources; otherwise use the whole clip.
- **Why:** a radio clip carries both voices; analysing the whole thing conflated them (e.g. the engineer's "Get in there Lewis!" was read as the driver's emotion). The in-car mic is loud and noisy (engine/wind, heavy compression) vs the clean pit-wall feed, so noisier utterances ≈ the driver.
- **Alternatives / trade-off:** **pyannote speaker diarization** was offered and *rejected by the user* — it is a gated model needing an HF token + accepting terms, is heavier, and still needs a heuristic to label which speaker is the driver. The energy heuristic is cruder but needs no gated model and is **conservative by design** (never touches single-speaker clips → no regression). It cannot isolate a driver who never speaks in a clip.

### 1.4 Show the driver's real tone + confidence
- **Decision:** the API returns `tone` (readable emotion) and `confidence` (0–100, the model's top probability); the UI shows "driver voice: angry · 74% sure".
- **Why:** the previous confidence was a fake number derived from the stress score. Showing the real detected emotion + real confidence is honest and more informative.

### 1.5 `mood.py` stays hand-written (not another model)
- **Decision:** the emotion→race-stress mapping (`calm/focused/stressed/tired` + 0–100) is deliberate rule-based domain logic.
- **Why:** no model maps *acted emotion* to *race* stress/fatigue; this is the project's value-add ("not a raw tool call") and the intended split — HF does perception, our code does the domain glue. Later laps skew distress toward fatigue.

---

## 2. Data & Hugging Face dataset

### 2.1 Use the HF dataset via the datasets-server API, not a full download
- **Decision:** pull clips through the datasets-server `/filter` and `/statistics` endpoints; download only the handful of audio files for the chosen driver+GP.
- **Why:** the dataset is 2.5 GB; we only ever need a few clips at a time. `/filter` returns the matching rows with short-lived signed mp3 URLs we fetch immediately.
- **Trade-off:** the datasets-server 500s / times out intermittently — mitigated with retries; import can be re-run.

### 2.2 Real lap alignment via FastF1 absolute lap times
- **Decision:** map each clip to the lap it was sent on by matching its UTC `message_timestamp` to FastF1's absolute `LapStartDate`.
- **Why:** "real FastF1 lap times, not toy numbers" is the project's research signal; guessing the lap would undercut it.
- **Trade-off:** `LapStartDate` is only populated when telemetry is loaded (`ses.load(telemetry=True)`) — a heavier load, but cached on disk after the first time.

### 2.3 Parallel clip downloads
- **Decision:** download a driver's clips concurrently with a `ThreadPoolExecutor` (8 workers).
- **Why:** the HF asset URLs are ~10 s each; sequential download of a full race took ~244 s. Parallel cut it to ~59 s (~4×).
- **Trade-off:** 8 concurrent requests; kept modest to avoid throttling.

### 2.4 Reject the mirror datasets (identical copies)
- **Decision:** stay on `MikCil/f1-team-radio` only; do **not** integrate `scriptaudio`, `fluffypotatoes`, `Tanishqbhatia`.
- **Why:** investigation showed all three are byte-identical copies (same 14,681 rows, same schema). A brief multi-mirror fallback was built for resilience, then removed at the user's direction — no different data to gain, not worth the complexity.

### 2.5 Cascading dropdowns built from the dataset (only show what has audio)
- **Decision:** Year → Grand Prix → (Session) → Driver/Team dropdowns are all populated from the dataset itself.
- **Why:** free-typing GP/driver names was error-prone (had to match the dataset exactly). Dataset-sourced options can only offer combinations that actually have audio. GP names were verified to match FastF1's event schedule 100%.

### 2.6 Dynamic session dropdown — only sessions with audio
- **Decision:** the dataset has no session field, so bucket each clip's timestamp into a FastF1 session (event schedule `Session{1..5}DateUtc`, nearest preceding start) and offer only sessions with clips; default to Race.
- **Why:** the Session dropdown previously always showed R/Q/S/FP1-3 even when a GP had no such audio. Now e.g. 2023 British shows only Qualifying + Race.
- **Trade-off:** requires paging the GP's rows + a FastF1 schedule load — both cached. Refactored so one cached fetch (`_gp_rows`) feeds both drivers and sessions.

### 2.7 Real team name from FastF1
- **Decision:** the dashboard's team comes from FastF1 (`driver_team_fastf1`), not a hardcoded value.
- **Why:** it previously showed "MERCEDES" for every driver (wrong for e.g. Ferrari's Leclerc). The FastF1 session is already loaded, so this is cheap.

---

## 3. Architecture & app shape

### 3.1 Live-only app (removed demo mode + offline fallback)
- **Decision:** deleted the offline demo endpoint (`/api/stint`), `demo_data.py`, and the frontend fallback; the app is the real pipeline end to end.
- **Why:** user directive. The app now needs models + FastF1 + clips to work.
- **Trade-off:** the original "judging-day safety net" is gone — an outright failure now shows an error rather than demo data. Flagged to the user, who accepted it.

### 3.2 CPU-only PyTorch
- **Decision:** install the CPU build of torch.
- **Why:** reliability on the target machine (older GPU/driver); a flaky GPU/CUDA download is a bigger demo risk than slower CPU inference.
- **Trade-off:** analysis takes tens of seconds per lap window on CPU.

### 3.3 Clips addressed by a `clips_dir` parameter (per-driver folders)
- **Decision:** import/analyze/clip-audio endpoints take a `clips_dir`; team mode uses `clips/<DRIVER>` per driver.
- **Why:** lets two drivers' clips coexist without collision for team mode, with no new endpoints — the parameter already existed.
- **Safety:** `_safe_clips_dir` keeps all paths inside the project.

### 3.4 Import clears prior clips; playback is cache-busted
- **Decision:** an import wipes the target clip dir first (`clear=True`); audio URLs carry `?v=<version>` bumped on each import/analysis.
- **Why:** switching driver/GP otherwise left stale clips (analysed against the wrong race) and the browser served the previous race's cached audio (same `/api/clips/5/audio` URL).

### 3.5 Frontend dashboard is a reusable component
- **Decision:** `makeDashboard(container, {laps, meta, clipsDir, audioVersion})` builds a self-contained panel (scoped DOM, own `<audio>`); `renderResults([...])` renders one panel (driver mode) or two (team mode).
- **Why:** team mode needs two independent dashboards on screen; the original single hard-coded instance couldn't do that.
- **Trade-off:** a sizable frontend refactor, but the clean way (vs suffixing element IDs).

### 3.6 Team mode layout — two separate dashboards
- **Decision:** in team mode, render each driver's **full** dashboard (chart, stats, radio, playback), stacked.
- **Why:** the user chose this over an overlaid head-to-head comparison chart — they wanted to read each driver's analysis independently.
- **Trade-off:** longer page + two analysis passes (~2× time); the most complete option.

---

## 4. Code quality

### 4.1 Prefer libraries over hand-rolled code
- **Decision:** use pandas for lap-time wrangling (window filter + NaT drop + sort) and CSV I/O; `concurrent.futures` for parallel downloads; removed dead code (unused imports `re`/`Optional`, dead frontend STATE fields).
- **Why:** shorter, more robust, and pandas was already a dependency (via fastf1). Directly requested.

### 4.2 Bug fixes worth recording
- FastF1 `lap_times_fastf1` dropped `NaT` laps (pit/in-out laps) that previously became `t=nan` rows.
- The mirror-fallback experiment was fully reverted (no dead code left).

---

## Open items / known limits
- **Emotion is a signal, not a verdict** — acted-emotion models are noisy on compressed radio; most F1 radio reads neutral/calm because drivers are composed.
- **Driver isolation is a heuristic**, not diarization (see 1.3).
- **Driver dropdown is GP-level, not session-level** — it lists everyone with radio at the GP, even if the chosen session has none for that specific driver. Could be made session-aware if needed.
- **datasets-server flakiness** — import can fail transiently (HTTP 500); re-run, or import early before a demo so clips are cached locally.
