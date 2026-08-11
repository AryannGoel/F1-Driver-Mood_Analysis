"""Emotion probabilities -> (mood, stress).

The HF speech-emotion models output generic acted-emotion classes (RAVDESS or
IEMOCAP). Race-engineering cares about *stress* and *fatigue*, not "surprised",
so this module is the domain layer that translates one into the other. It's the
part that isn't a raw tool call -- exactly the "balanced difficulty" the brief asks for.

Supported label vocabularies (case-insensitive):
  RAVDESS-8       : angry calm disgust fearful happy neutral sad surprised
  IEMOCAP-4 (SER) : ang  neu  hap  sad
  text-emotion    : anger disgust fear joy neutral sadness surprise
"""

# map any model label -> one canonical bucket
_ALIAS = {
    "ang": "angry", "angry": "angry", "anger": "angry",
    "fea": "fearful", "fearful": "fearful", "fear": "fearful",
    "dis": "disgust", "disgust": "disgust",
    "sur": "surprised", "surprised": "surprised", "surprise": "surprised",
    "sad": "sad", "sadness": "sad",
    "hap": "happy", "happy": "happy", "joy": "happy",
    "neu": "neutral", "neutral": "neutral",
    "cal": "calm", "calm": "calm",
}
_CANON = ["angry", "fearful", "disgust", "surprised", "sad", "happy", "neutral", "calm"]

# how much each emotion contributes to a "stress" reading
_W = {"angry": 0.90, "fearful": 1.00, "disgust": 0.70,
      "surprised": 0.55, "sad": 0.50, "happy": 0.0, "neutral": 0.0, "calm": 0.0}


def _canonical(probs: dict) -> dict:
    out = {k: 0.0 for k in _CANON}
    for label, score in probs.items():
        key = _ALIAS.get(str(label).strip().lower())
        if key:
            out[key] += float(score)
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}


def fuse(probs: dict, text_probs: dict = None, text_weight: float = 0.4) -> dict:
    """Blend the acoustic emotion (tone of voice) with the linguistic emotion (the
    words), both mapped to the canonical buckets. Team radio is short/compressed so
    the voice model is often unsure; the transcript's emotion steadies the read."""
    p = _canonical(probs)
    if text_probs:
        pt = _canonical(text_probs)
        p = {k: (1 - text_weight) * p[k] + text_weight * pt[k] for k in _CANON}
        total = sum(p.values()) or 1.0
        p = {k: v / total for k, v in p.items()}
    return p


def classify(probs: dict, progress: float = 0.0, text_probs: dict = None) -> tuple[str, int]:
    """probs: {emotion_label: score} from the speech-emotion model.
    text_probs: optional {emotion_label: score} from the transcript emotion model.
    progress: 0..1 position within the stint (later laps skew a distress reading
    toward fatigue rather than acute stress)."""
    p = fuse(probs, text_probs)
    top = max(p, key=p.get)

    # arousal/distress score. sad counts too -- a flat, spent voice is still a warning.
    raw = sum(_W[k] * p[k] for k in _CANON)
    stress = int(max(8, min(98, round(raw * 100))))

    sad_dom = p["sad"] >= 0.40 and top == "sad"
    fatigue = p["sad"] + 0.4 * p["neutral"]

    if sad_dom and progress > 0.40:
        mood = "tired"          # spent voice, late in the stint = fatigue
    elif stress >= 45 and progress > 0.55 and fatigue > 0.35:
        mood = "tired"          # sustained distress deep in the stint
    elif stress >= 45:
        mood = "stressed"       # acute, high-arousal distress
    elif stress >= 25 or sad_dom:
        mood = "focused"        # elevated but controlled
    else:
        mood = "calm"
    return mood, stress


if __name__ == "__main__":
    # quick sanity check with hand-made distributions
    calm = {"calm": 0.7, "neutral": 0.3}
    panic = {"fearful": 0.6, "angry": 0.3, "neutral": 0.1}
    spent = {"sad": 0.6, "neutral": 0.4}
    print("calm  ->", classify(calm, 0.1))
    print("panic ->", classify(panic, 0.4))
    print("spent ->", classify(spent, 0.8))
