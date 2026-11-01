"""Korean narration.

Engines (TTS_ENGINE):
  google  Google Cloud Text-to-Speech (commercial). Needs the env var GOOGLE_TTS_API_KEY
          (an API key with the Text-to-Speech API enabled). Default voice: a Korean male
          Chirp 3 HD voice, override with TTS_VOICE (e.g. ko-KR-Chirp3-HD-Charon,
          ko-KR-Neural2-C, ko-KR-Wavenet-D).
  vits    offline VITS voice via sherpa-onnx (female KSS). Several takes per line, the one
          the SenseVoice ASR transcribes best is kept.

Fit mode (TTS_FIT=1): instead of letting the voice decide the timeline, every line is
time-stretched (rubberband, pitch preserved) to the voiceDur already stored in
build/timeline.json, so a new voice drops into already rendered frames with the karaoke
subtitles still in sync.

Writes build/voice/<id>.wav (48 kHz mono), durations.json and qa.json.
"""
import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "build", "models")
TTS_DIR = os.environ.get("KO_TTS_MODEL", os.path.join(MODELS, "vits-mimic3-ko_KO-kss_low"))
ASR_DIR = os.environ.get("KO_ASR_MODEL", os.path.join(MODELS, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"))
OUT = os.path.join(ROOT, "build", "voice")
SR = 48000

ENGINE = os.environ.get("TTS_ENGINE", "google" if os.environ.get("GOOGLE_TTS_API_KEY") else "vits")
VOICE = os.environ.get("TTS_VOICE", "ko-KR-Chirp3-HD-Achird")
FIT = os.environ.get("TTS_FIT") == "1"


# ---------------------------------------------------------------- engines

class GoogleTTS:
    URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

    def __init__(self, voice):
        import requests
        self.requests = requests
        self.key = os.environ["GOOGLE_TTS_API_KEY"]
        self.voice = voice
        self.rate_ok = True

    def synth(self, text, rate=1.0):
        body = {
            "input": {"text": text},
            "voice": {"languageCode": "ko-KR", "name": self.voice},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": SR},
        }
        if self.rate_ok and abs(rate - 1.0) > 1e-3:
            body["audioConfig"]["speakingRate"] = round(rate, 3)
        r = self.requests.post(self.URL, params={"key": self.key}, json=body, timeout=60)
        if r.status_code == 400 and "speakingRate" in body["audioConfig"] and "peaking" in r.text:
            self.rate_ok = False  # voice without rate control: fall back to time-stretching
            return self.synth(text, 1.0)
        if r.status_code != 200:
            raise RuntimeError(f"Google TTS {r.status_code}: {r.text[:300]}")
        x, sr = sf.read(io.BytesIO(base64.b64decode(r.json()["audioContent"])), dtype="float64")
        if x.ndim > 1:
            x = x.mean(1)
        return x, sr


class VitsTTS:
    def __init__(self):
        import sherpa_onnx
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=os.path.join(TTS_DIR, "ko_KO-kss_low.onnx"),
            tokens=os.path.join(TTS_DIR, "tokens.txt"),
            data_dir=os.path.join(TTS_DIR, "espeak-ng-data"),
            noise_scale=0.55, noise_scale_w=0.6, length_scale=1.0,
        )
        self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=4)))

    def synth(self, text, rate=1.0):
        # the phonemizer drops ! and ?, so end every clause on a full stop for a clean final syllable
        a = self.tts.generate(text.replace("!", ".").replace("?", "."), sid=0, speed=rate)
        return np.asarray(a.samples, dtype=np.float64), a.sample_rate


# ---------------------------------------------------------------- QA

def load_asr():
    model = os.path.join(ASR_DIR, "model.int8.onnx")
    if not os.path.exists(model):
        return None
    import sherpa_onnx
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model, tokens=os.path.join(ASR_DIR, "tokens.txt"), language="ko", use_itn=False, num_threads=4)


def transcribe(asr, x, sr):
    s = asr.create_stream()
    s.accept_waveform(16000, signal.resample_poly(x, 16000, sr).astype(np.float32))
    asr.decode_stream(s)
    return s.result.text


def cer(ref, hyp):
    norm = lambda s: re.sub(r"[^가-힣a-zA-Z0-9]", "", s)
    ref, hyp = norm(ref), norm(hyp)
    d = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[len(hyp)] / max(1, len(ref))


# ---------------------------------------------------------------- processing

def trim(x, sr, pad_a=0.03, pad_b=0.12):
    thresh = 0.01 * np.max(np.abs(x))
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x
    y = x[max(0, idx[0] - int(pad_a * sr)): min(len(x), idx[-1] + int(pad_b * sr))].copy()
    fade = int(0.03 * sr)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y


def stretch(x, factor):
    """Pitch-preserving time stretch: output length = len(x) / factor."""
    if abs(factor - 1) < 0.004:
        return x
    with tempfile.TemporaryDirectory() as d:
        a, b = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
        sf.write(a, x.astype(np.float32), SR)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-af",
                        f"rubberband=tempo={factor:.5f}:pitchq=quality:formant=preserved", b], check=True)
        y, _ = sf.read(b, dtype="float64")
    return y


def polish(x, studio):
    """Voice chain. Studio voices only get cleanup; the offline voice gets more shaping."""
    x = signal.sosfilt(signal.butter(2, 75 if studio else 90, "highpass", fs=SR, output="sos"), x)
    lifts = ((180, 0.9, 0.12), (3000, 1.4, 0.12)) if studio else ((3200, 1.2, 0.3), (220, 0.9, 0.15))
    for f0, q, g in lifts:
        b, a = signal.iirpeak(f0, q, fs=SR)
        x = x + g * signal.lfilter(b, a, x)
    # de-ess: tame 5-9 kHz when it spikes
    s = signal.sosfilt(signal.butter(2, [5000, 9000], "bandpass", fs=SR, output="sos"), x)
    se = np.sqrt(signal.lfilter([0.01], [1, -0.99], s * s) + 1e-12)
    xe = np.sqrt(signal.lfilter([0.01], [1, -0.99], x * x) + 1e-12)
    x = x - s * np.clip(1 - 0.35 * xe / (se + 1e-9), 0, 0.6)
    if not studio:
        x = signal.sosfilt(signal.butter(2, 10500, "lowpass", fs=SR, output="sos"), x)
    env = np.sqrt(signal.lfilter([0.002], [1, -0.998], x * x) + 1e-9)
    thr, ratio = (0.1, 2.5) if studio else (0.08, 3.0)
    x = x * np.where(env > thr, (thr * (env / thr) ** (1 / ratio)) / env, 1.0)
    return x / (np.max(np.abs(x)) + 1e-9) * 0.89


# ---------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    lines = json.load(open(os.path.join(ROOT, "audio", "narration.json"), encoding="utf-8"))
    targets = {}
    if FIT:
        tl = json.load(open(os.path.join(ROOT, "build", "timeline.json")))
        targets = {s["id"]: s["voiceDur"] for s in tl["scenes"]}
    engine = GoogleTTS(VOICE) if ENGINE == "google" else VitsTTS()
    studio = ENGINE == "google"
    asr = load_asr()
    print(f"engine={ENGINE}" + (f" voice={VOICE}" if studio else "") + (" (fit to timeline)" if FIT else ""), file=sys.stderr)

    durations, qa = {}, {}
    for line in lines:
        text = line.get("spoken", line["text"])
        target = targets.get(line["id"])
        takes = [] if studio else [(r, k) for r in (0.94, 1.0, 1.04) for k in range(3 if asr else 1)]

        if studio:
            raw, sr = engine.synth(text, 1.0)
            x = trim(signal.resample_poly(raw, SR, sr) if sr != SR else raw, SR)
            if target:
                # get close with the voice's own speaking rate, then fine-tune by stretching
                rate = float(np.clip((len(x) / SR) / target, 0.85, 1.25))
                if abs(rate - 1) > 0.03:
                    raw, sr = engine.synth(text, rate)
                    x = trim(signal.resample_poly(raw, SR, sr) if sr != SR else raw, SR)
            hyp = transcribe(asr, x, SR) if asr else ""
            score = cer(text, hyp) if asr else 0.0
        else:
            best = None
            for rate, _ in takes:
                raw, sr = engine.synth(text, rate)
                hyp = transcribe(asr, raw, sr) if asr else ""
                score = (cer(text, hyp) + abs(rate - 1.0) * 0.2) if asr else 0.0
                if best is None or score < best[0]:
                    best = (score, raw, sr, hyp)
            score, raw, sr, hyp = best
            x = trim(signal.resample_poly(raw, SR, sr), SR)

        if target:
            factor = (len(x) / SR) / target
            if not 0.75 < factor < 1.35:
                print(f"  ! {line['id']}: needs {factor:.2f}x stretch to fit; consider re-timing", file=sys.stderr)
            x = stretch(x, factor)
            x = np.pad(x, (0, max(0, int(target * SR) - len(x))))[: int(target * SR)]
        x = polish(x, studio)
        sf.write(os.path.join(OUT, f"{line['id']}.wav"), x.astype(np.float32), SR)
        durations[line["id"]] = round(len(x) / SR, 3)
        qa[line["id"]] = {"text": line["text"], "asr": hyp, "cer": round(score, 3)}
        print(f"{line['id']:9s} {durations[line['id']]:.2f}s  cer={score:.2f}  asr: {hyp}", file=sys.stderr)

    json.dump(durations, open(os.path.join(OUT, "durations.json"), "w"), indent=2)
    json.dump({"engine": ENGINE, "voice": VOICE if studio else "kss", "lines": qa}, open(os.path.join(OUT, "qa.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
