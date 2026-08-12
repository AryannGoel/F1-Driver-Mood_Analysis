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
#
# Audio is loaded with soundfile + soxr and framed with plain NumPy — deliberately
# no librosa, because librosa pulls in numba, whose compiled extension is blocked on
# some locked-down Windows machines (Application Control), which used to crash the
# whole analysis at the first clip.
_HOP = 512
_FRAME = 2048


def _load_audio(path, sr=16000):
    """Decode any clip to a mono float32 array at `sr` Hz (no librosa/numba)."""
    import numpy as np
    import soundfile as sf
    y, orig_sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:                     # stereo -> mono
        y = y.mean(axis=1)
    y = np.ascontiguousarray(y, dtype="float32")
    if orig_sr != sr and len(y):
        try:
            import soxr
            y = soxr.resample(y, orig_sr, sr).astype("float32")
        except Exception:                             # linear-interp fallback
            n = max(1, int(round(len(y) * sr / orig_sr)))
            y = np.interp(np.linspace(0, len(y), n, endpoint=False),
                          np.arange(len(y)), y).astype("float32")
    return y, sr


def _frames(y):
    """Centre-padded frames (frame=_FRAME, hop=_HOP), matching librosa's framing so
    the RMS and flatness arrays line up frame-for-frame with the old behaviour."""
    import numpy as np
    pad = _FRAME // 2
    yp = np.pad(y, pad, mode="reflect") if len(y) > pad else np.pad(y, pad)
    n = 1 + (len(yp) - _FRAME) // _HOP
    if n <= 0:
        return np.empty((0, _FRAME), dtype="float32")
    idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(n)[:, None]
    return yp[idx]


def _rms(y):
    """Per-frame root-mean-square energy (numpy reimplementation)."""
    import numpy as np
    fr = _frames(y)
    if not len(fr):
        return np.zeros(1, dtype="float32")
    return np.sqrt(np.mean(fr * fr, axis=1)).astype("float32")


def _spectral_flatness(y):
    """Per-frame spectral flatness = geometric/arithmetic mean of the power
    spectrum, in (0, 1] (numpy reimplementation of librosa.feature.spectral_flatness)."""
    import numpy as np
    fr = _frames(y)
    if not len(fr):
        return np.zeros(1, dtype="float32")
    power = np.abs(np.fft.rfft(fr * np.hanning(_FRAME), axis=1)) ** 2
    power = np.maximum(power, 1e-10)
    gmean = np.exp(np.mean(np.log(power), axis=1))
    amean = np.mean(power, axis=1)
    return (gmean / amean).astype("float32")


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
    y, sr = _load_audio(path)
    if len(y) < int(1.2 * sr):
        return y
    rms = _rms(y)
    flat = _spectral_flatness(y)
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
