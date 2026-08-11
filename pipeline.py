"""The genuine Hugging Face work: two Hub models doing real inference.

  ASR : openai/whisper-base.en          -> radio audio to transcript
  SER : superb/wav2vec2-base-superb-er  -> audio to emotion probabilities

Both are loaded lazily and cached, so importing this module is free; the models
only download the first time you actually run live mode. Override either via env:

  ASR_MODEL=openai/whisper-small.en
  SER_MODEL=ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition

Note on the SER model: the popular `ehcalabres/wav2vec2-lg-xlsr-...-emotion`
checkpoint no longer loads its classifier head under transformers 5.x (the head
params are missing from the checkpoint, so they get randomly initialised and the
output is a meaningless near-uniform distribution). `superb/wav2vec2-base-superb-er`
is an official SUPERB model whose trained head loads cleanly and gives confident,
discriminating emotions (IEMOCAP-4: angry/happy/neutral/sad) that mood.py maps.
"""
import os
import functools
from mood import classify, fuse

ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-base.en")
SER_MODEL = os.environ.get("SER_MODEL", "superb/wav2vec2-base-superb-er")
# text-emotion on the transcript, fused with the voice tone for a steadier read
TXT_EMO_MODEL = os.environ.get("TXT_EMO_MODEL", "j-hartmann/emotion-english-distilroberta-base")


@functools.lru_cache(maxsize=1)
def _asr():
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=ASR_MODEL)


@functools.lru_cache(maxsize=1)
def _ser():
    from transformers import pipeline
    # top_k=None returns scores for every class, which mood.classify needs
    return pipeline("audio-classification", model=SER_MODEL, top_k=None)


@functools.lru_cache(maxsize=1)
def _text_emotion():
    from transformers import pipeline
    return pipeline("text-classification", model=TXT_EMO_MODEL, top_k=None)


def text_emotions(text: str) -> dict:
    """Emotion probabilities from the transcript, or {} if there's no text."""
    text = (text or "").strip()
    if not text:
        return {}
    preds = _text_emotion()(text[:512])          # truncate very long exchanges
    if preds and isinstance(preds[0], list):     # top_k=None can nest one level
        preds = preds[0]
    return {p["label"]: float(p["score"]) for p in preds}


# ---- isolate the DRIVER's voice from the race engineer's --------------------
# Team radio carries both the driver (in-car mic: loud, noisy — engine/wind, heavy
# compression) and the pit-wall engineer (clean feed). To analyse the *driver* and
# not the team, we keep the noisier (in-car) speech and drop the clean engineer
# speech — but only when the clip clearly splits into two sources; a single-speaker
# clip is left whole, so this never degrades the common case.
_HOP = 512


def _load_audio(path, sr=16000):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype("float32"), sr


def _utterances(voiced, sr):
    """Merge voiced frames (gaps < 0.4s bridged) into sentence-level utterances."""
    n = len(voiced)
    gap = int(0.4 * sr / _HOP)
    min_len = int(0.4 * sr / _HOP)
    utts, i = [], 0
    while i < n:
        if voiced[i]:
            j, last = i, i
            while j < n:
                if voiced[j]:
                    last = j; j += 1
                elif (j - last) <= gap:
                    j += 1
                else:
                    break
            utts.append((i, last + 1)); i = j
        else:
            i += 1
    return [(s, e) for (s, e) in utts if (e - s) >= min_len]


def driver_audio(path):
    """Load a clip and return the driver's (in-car) audio as a 16 kHz array."""
    import numpy as np
    import librosa
    y, sr = _load_audio(path)
    if len(y) < int(1.2 * sr):
        return y
    rms = librosa.feature.rms(y=y, hop_length=_HOP)[0]
    flat = librosa.feature.spectral_flatness(y=y, hop_length=_HOP)[0]
    if rms.max() <= 0:
        return y
    thr = max(float(np.median(rms)) * 1.5, float(rms.max()) * 0.2)
    utts = _utterances(rms > thr, sr)
    if len(utts) <= 1:
        return y                                   # single speaker -> whole clip
    flats = [float(flat[s:e].mean()) for s, e in utts]
    lo, hi = min(flats), max(flats)
    if hi < 1.8 * (lo + 1e-9):
        return y                                   # not clearly two sources
    cut = (lo + hi) / 2
    keep = [utts[k] for k, fl in enumerate(flats) if fl >= cut]   # noisier = in-car
    if not keep or len(keep) == len(utts):
        return y
    pad = int(0.08 * sr)
    return np.concatenate([y[max(0, s * _HOP - pad):e * _HOP + pad] for s, e in keep])


def transcribe(audio) -> str:
    """audio: a file path or a raw 16 kHz numpy array."""
    out = _asr()(audio)
    return (out.get("text") or "").strip()


def emotions(audio) -> dict:
    """-> {label: score} across all emotion classes. audio: path or 16 kHz array."""
    preds = _ser()(audio)
    return {p["label"]: float(p["score"]) for p in preds}


# readable names for the raw SER emotion classes (superb uses short IEMOCAP codes)
_TONE_LABEL = {"ang": "angry", "hap": "happy", "neu": "neutral", "sad": "sad",
               "angry": "angry", "happy": "happy", "neutral": "neutral",
               "calm": "calm", "fearful": "fearful", "disgust": "disgust",
               "surprised": "surprised"}


def analyse_clip(path: str, lap: int, lap_time: float, progress: float) -> dict:
    """One radio clip -> one lap row. Transcript + tone come from the driver's
    isolated voice, so the engineer's calm delivery doesn't skew the reading.
    `tone`/`confidence` expose the driver's actual detected emotion + how sure the
    model is, rather than a derived number."""
    driver = driver_audio(path)
    text = transcribe(driver)
    voice_emo = emotions(driver)                    # tone of voice  (HF: wav2vec2)
    text_emo = text_emotions(text)                  # emotion in words (HF: distilroberta)
    mood, stress = classify(voice_emo, progress, text_probs=text_emo)
    # report the driver's dominant emotion from the fused (voice + words) read
    fused = fuse(voice_emo, text_emo)
    top = max(fused, key=fused.get) if fused else ""
    tone = _TONE_LABEL.get(str(top).lower(), str(top).lower())
    confidence = int(round(float(fused.get(top, 0.0)) * 100)) if fused else 0
    return {"lap": lap, "t": round(lap_time, 1), "stress": stress, "mood": mood,
            "radio": text, "tone": tone, "confidence": confidence}
