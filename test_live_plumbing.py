"""Verify the LIVE code path (schema + merge + progress + no-radio baseline)
with the two HF model calls and FastF1 stubbed out. Proves everything works
except the actual model inference (which needs real deps + audio on your machine)."""
import laps, pipeline
from fastapi.testclient import TestClient
from app import app

# --- stub the parts that need torch/fastf1/network ---
laps.lap_times_fastf1 = lambda *a, **k: [{"lap": n, "t": 91.0 + n*0.1} for n in range(28, 43)]
laps.local_clips = lambda folder="clips": {34: "fake34.wav", 37: "fake37.wav"}
def fake_analyse(path, lap, t, progress):
    mood = "stressed" if progress < 0.5 else "tired"
    return {"lap": lap, "t": round(t,1), "stress": 65+lap%10, "mood": mood, "radio": f"clip for lap {lap}"}
pipeline.analyse_clip = fake_analyse

c = TestClient(app)
r = c.post("/api/analyze", json={"year":2023,"gp":"X","session":"R","driver":"VER","lap_from":28,"lap_to":42})
print("status:", r.status_code)
d = r.json()
assert r.status_code == 200, d
assert d["meta"]["source"] == "live"
assert len(d["laps"]) == 15
radio_lap  = next(l for l in d["laps"] if l["lap"] == 34)
silent_lap = next(l for l in d["laps"] if l["lap"] == 35)
assert radio_lap["radio"] and radio_lap["stress"] > 0, radio_lap
assert silent_lap["radio"] is None and silent_lap["mood"] == "calm", silent_lap
print("analysed radio lap :", radio_lap)
print("silent lap baseline:", silent_lap)
print("PASS - live plumbing + schema validation OK")
