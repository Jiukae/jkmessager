"""Synthesize the Korean narration offline with a VITS voice (sherpa-onnx).

Several takes are generated per line (different speeds / sampling noise); if the
SenseVoice ASR model is available, each take is transcribed and the one whose
transcript matches the script best is kept.

Writes build/voice/<id>.wav (48 kHz mono, processed), build/voice/durations.json
(the timeline is derived from it) and build/voice/qa.json (ASR transcripts).
"""
import json
import os
import re
import sys

import numpy as np
import sherpa_onnx
import soundfile as sf
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "build", "models")
TTS_DIR = os.environ.get("KO_TTS_MODEL", os.path.join(MODELS, "vits-mimic3-ko_KO-kss_low"))
ASR_DIR = os.environ.get("KO_ASR_MODEL", os.path.join(MODELS, "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"))
OUT = os.path.join(ROOT, "build", "voice")
SR = 48000
SPEEDS = (0.94, 1.0, 1.04)
TRIES = 3


def load_tts():
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=os.path.join(TTS_DIR, "ko_KO-kss_low.onnx"),
        tokens=os.path.join(TTS_DIR, "tokens.txt"),
        data_dir=os.path.join(TTS_DIR, "espeak-ng-data"),
        noise_scale=0.55,
        noise_scale_w=0.6,
        length_scale=1.0,
    )
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=4))
    return sherpa_onnx.OfflineTts(cfg)


def load_asr():
    model = os.path.join(ASR_DIR, "model.int8.onnx")
    if not os.path.exists(model):
        print("ASR model not found; keeping the first take of each line", file=sys.stderr)
        return None
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model, tokens=os.path.join(ASR_DIR, "tokens.txt"), language="ko", use_itn=False, num_threads=4)


def transcribe(asr, x, sr):
    y = signal.resample_poly(x, 16000, sr).astype(np.float32)
    s = asr.create_stream()
    s.accept_waveform(16000, y)
    asr.decode_stream(s)
    return s.result.text


def norm(s):
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", s)


def cer(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    d = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
    return d[len(hyp)] / max(1, len(ref))


def trim(x, thresh, pad_a=0.03, pad_b=0.12, sr=SR):
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(pad_a * sr))
    b = min(len(x), idx[-1] + int(pad_b * sr))
    y = x[a:b].copy()
    fade = int(0.03 * sr)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y


def polish(x, sr=SR):
    """Voice chain: HPF, presence/warmth lift, LPF, compression, peak normalise."""
    x = signal.sosfilt(signal.butter(2, 90, "highpass", fs=sr, output="sos"), x)
    for f0, q, g in ((3200, 1.2, 0.3), (220, 0.9, 0.15)):
        b, a = signal.iirpeak(f0, q, fs=sr)
        x = x + g * signal.lfilter(b, a, x)
    x = signal.sosfilt(signal.butter(2, 10500, "lowpass", fs=sr, output="sos"), x)
    env = np.sqrt(signal.lfilter([0.002], [1, -0.998], x * x) + 1e-9)
    thr, ratio = 0.08, 3.0
    gain = np.where(env > thr, (thr * (env / thr) ** (1 / ratio)) / env, 1.0)
    x = x * gain
    return x / (np.max(np.abs(x)) + 1e-9) * 0.89


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = json.load(open(os.path.join(ROOT, "audio", "narration.json"), encoding="utf-8"))
    tts, asr = load_tts(), load_asr()
    durations, qa = {}, {}
    for line in lines:
        # the phonemizer drops ! and ?, so end every clause on a full stop for a clean final syllable
        spoken = line.get("spoken", line["text"]).replace("!", ".").replace("?", ".")
        best = None
        for speed in SPEEDS:
            for _ in range(TRIES if asr else 1):
                audio = tts.generate(spoken, sid=0, speed=speed)
                raw = np.asarray(audio.samples, dtype=np.float64)
                score, hyp = 0.0, ""
                if asr:
                    hyp = transcribe(asr, raw, audio.sample_rate)
                    score = cer(spoken, hyp) + abs(speed - 1.0) * 0.2
                if best is None or score < best[0]:
                    best = (score, raw, audio.sample_rate, hyp, speed)
            if not asr:
                break
        score, raw, sr_in, hyp, speed = best
        x = signal.resample_poly(raw, SR, sr_in)
        x = polish(trim(x, thresh=0.01 * np.max(np.abs(x))))
        sf.write(os.path.join(OUT, f"{line['id']}.wav"), x.astype(np.float32), SR)
        durations[line["id"]] = round(len(x) / SR, 3)
        qa[line["id"]] = {"text": line["text"], "asr": hyp, "score": round(score, 3), "speed": speed}
        print(f"{line['id']:9s} {durations[line['id']]:.2f}s  score={score:.2f}  asr: {hyp}", file=sys.stderr)
    json.dump(durations, open(os.path.join(OUT, "durations.json"), "w"), indent=2)
    json.dump(qa, open(os.path.join(OUT, "qa.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
