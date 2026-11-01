"""Procedural sound design + lo-fi music + final mix (no sample libraries).

Inputs : build/timeline.json, build/events.json (physics contacts + cue times), build/voice/*.wav
Output : build/audio/mix.wav (48 kHz stereo) and the stems in build/audio/

Every ice clink is placed on a real rigid-body contact from the simulation, its
loudness follows the impact speed, and its pitch/damping follow how full the
glass is at that moment (liquid lowers and deadens a glass's ring).
"""
import json
import os

import numpy as np
import soundfile as sf
from scipy import signal

import sfx as sfx_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.path.join(ROOT, "build")
OUT = os.path.join(B, "audio")
SR = 48000
rng = np.random.default_rng(1234)

tl = json.load(open(os.path.join(B, "timeline.json")))
ev = json.load(open(os.path.join(B, "events.json")))
S = {s["id"]: s for s in tl["scenes"]}
cues = ev["cues"]
DUR = tl["duration"] + 0.6
N = int(DUR * SR)


def buf():
    return np.zeros((N, 2))


def tvec(n):
    return np.arange(n) / SR


def place(bus, sig, t, gain=1.0, pan=0.0):
    """Add a mono or stereo signal at time t (s) with constant-power pan (-1..1)."""
    i = int(round(t * SR))
    if i >= N or gain == 0:
        return
    if i < 0:
        sig = sig[-i:]
        i = 0
    n = min(len(sig), N - i)
    if sig.ndim == 1:
        a = (pan + 1) * np.pi / 4
        bus[i:i + n, 0] += sig[:n] * gain * np.cos(a)
        bus[i:i + n, 1] += sig[:n] * gain * np.sin(a)
    else:
        bus[i:i + n] += sig[:n] * gain


def bp(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x, axis=0)


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "lowpass", fs=SR, output="sos"), x, axis=0)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "highpass", fs=SR, output="sos"), x, axis=0)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def flow_progress(t, t0, t1, ramp):
    total = t1 - t0 - ramp
    t = np.asarray(t, dtype=float)
    a = np.where(t < t0 + ramp, (t - t0) ** 2 / (2 * ramp),
                 np.where(t <= t1 - ramp, ramp / 2 + (t - t0 - ramp), total - (t1 - t) ** 2 / (2 * ramp)))
    return np.clip(np.where(t <= t0, 0, np.where(t >= t1, total, a)) / total, 0, 1)


def flow_rate(t, t0, t1, ramp):
    return smoothstep(t0, t0 + ramp, t) * (1 - smoothstep(t1 - ramp, t1, t))


BASE, LW, LF, H = 1.1, 10.3, 12.1, 14.0
W, P = cues["water"], cues["pour"]


def level(t):
    w = flow_progress(t, W["t0"], W["t1"], 0.35)
    p = flow_progress(t, P["t0"], P["t1"], 0.4)
    lv = BASE + (LW - BASE) * w + (LF - LW) * p
    return np.where(np.asarray(t) < W["t0"], BASE, lv)


def air_column(lv):
    """Quarter-wave resonance of the air above the liquid (with end correction)."""
    return 343 / (4 * (H - lv + 2.3) / 100)


WALLA_LINES = [
    "오늘 날씨 진짜 좋다", "이거 생각보다 맛있네", "주말에 뭐 할 거야", "아 그래서 어떻게 됐어",
    "라떼 한 잔 더 시킬까", "여기 분위기 괜찮다", "케이크도 하나 시키자", "그 얘기 들었어",
    "완전 웃기지 않아", "나 요즘 운동 시작했어", "회의가 너무 길었어", "다음에 또 오자",
]


def make_walla():
    """Café murmur: overlapping far-away speakers made with the offline voice, then
    pitch-varied, band-limited and pushed back into the room until no word is legible."""
    cache = os.path.join(OUT, "walla.wav")
    if os.path.exists(cache):
        x, _ = sf.read(cache)
        return np.pad(x, ((0, max(0, N - len(x))), (0, 0)))[:N]
    model = os.path.join(B, "models", "vits-mimic3-ko_KO-kss_low")
    if not os.path.isdir(model):
        return None
    import sherpa_onnx
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(model=f"{model}/ko_KO-kss_low.onnx", tokens=f"{model}/tokens.txt", data_dir=f"{model}/espeak-ng-data")
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=4)))
    clips = []
    for text in WALLA_LINES:
        a = tts.generate(text, sid=0, speed=1.1)
        clips.append(signal.resample_poly(np.asarray(a.samples), SR, a.sample_rate))
    out = np.zeros((N, 2))
    for spk in range(6):
        pitch = rng.uniform(0.72, 1.12)  # <1 lowers the voice: several "people"
        pan = rng.uniform(-0.9, 0.9)
        t = rng.uniform(0, 1.5)
        while t < DUR:
            c = clips[rng.integers(len(clips))]
            c = signal.resample(c, int(len(c) / pitch))
            i = int(t * SR)
            n = min(len(c), N - i)
            if n <= 0:
                break
            a = (pan + 1) * np.pi / 4
            out[i:i + n, 0] += c[:n] * np.cos(a) * rng.uniform(0.5, 1.0)
            out[i:i + n, 1] += c[:n] * np.sin(a) * rng.uniform(0.5, 1.0)
            t += len(c) / SR + rng.uniform(0.3, 1.6)
    out = bp(out, 220, 2200)
    out = np.stack([signal.fftconvolve(out[:, c], sfx_mod.room_ir(0.9, 4000)[:, c])[:N] for c in range(2)], 1) * 0.5 + out * 0.15
    out /= np.max(np.abs(out)) + 1e-9
    os.makedirs(OUT, exist_ok=True)
    sf.write(cache, out.astype(np.float32), SR)
    return out


def visible_main(t):
    """1 while the glass is on screen (the espresso scene cuts to the machine)."""
    e = S["espresso"]
    return 0.0 if e["start"] + 0.05 < t < e["end"] - 0.05 else 1.0


# ---------------------------------------------------------------- SFX bus

sfx_mod.seed(1234)
sfx = buf()
water_pan, pour_pan = 0.28, 0.25  # both the carafe and the shot glass come in from the right

# physics contacts -> impacts (the first cubes land on the glass foot)
first_glass = True
for e in ev["events"]:
    t = e["t"]
    if visible_main(t) == 0:
        continue
    fill = float(np.clip((e["level"] - BASE) / (H - BASE), 0, 1)) if e["level"] > 0 else 0.0
    pan = float(np.clip(rng.normal(0, 0.16), -0.45, 0.45))
    if e["kind"] == "glass":
        low = e["y"] < 3.5
        place(sfx, sfx_mod.glass_hit(e["v"], fill, e["under"], low=low), t, 0.55, pan)
    elif e["kind"] == "straw":
        place(sfx, sfx_mod.straw_hit(e["v"], e["under"]), t, 0.4, pan)
    else:
        place(sfx, sfx_mod.ice_hit(e["v"], e["under"]), t, 0.34, pan)

# hero shots: lazy clinks of settling ice in a full glass
for t, v in ((0.75, 45), (2.3, 30), (3.85, 38), (S["outro"]["start"] + 0.9, 40), (S["outro"]["start"] + 2.6, 28)):
    place(sfx, sfx_mod.glass_hit(v, 0.85, False), t, 0.4, rng.uniform(-0.2, 0.2))
    place(sfx, sfx_mod.ice_hit(v * 0.7, False), t + rng.uniform(0.03, 0.08), 0.22, rng.uniform(-0.2, 0.2))

column = lambda tt: air_column(float(level(tt)))

# cold water from the carafe
w_rate = lambda t: flow_rate(t, W["t0"], W["t1"], 0.35)
place(sfx, sfx_mod.pour(W["t0"], W["t1"], w_rate, column, stream=1.0), W["t0"], 0.27, water_pan)
for _ in range(16):  # floating cubes nudging each other and the glass while the water churns
    t = rng.uniform(W["t0"] + 0.4, W["t1"] + 0.6)
    f = float((level(t) - BASE) / (H - BASE))
    hit = sfx_mod.glass_hit(rng.uniform(15, 40), f, True) if rng.random() < 0.5 else sfx_mod.ice_hit(rng.uniform(20, 60), rng.random() < 0.6)
    place(sfx, hit, t, 0.4, rng.uniform(-0.25, 0.35))
place(sfx, sfx_mod.crackle(2.5, lambda tt: 10 * np.exp(-tt / 1.0) + 1), W["t0"] + 0.2, 0.25, water_pan)

# espresso machine
E = cues["espresso"]
e0, e1 = E["pumpOn"], E["pumpOff"]
place(sfx, sfx_mod.rocker_click(), e0 - 0.16, 0.28, -0.15)
place(sfx, sfx_mod.solenoid(), e0, 0.25, -0.05)
place(sfx, sfx_mod.vibratory_pump(e1 - e0), e0, 0.1, -0.1)
place(sfx, sfx_mod.solenoid(release=True), e1, 0.28, 0.1)
place(sfx, sfx_mod.rocker_click(), e1 - 0.05, 0.22, -0.15)
shot_col = lambda tt: 343 / (4 * (6.5 - 0.7 - 2.35 * float(flow_progress(tt, E["flow0"] - 0.2, E["flow1"] + 0.2, 0.5)) + 1.6) / 100)
fl = lambda t: flow_rate(t, E["flow0"], E["flow1"], 0.5)
place(sfx, sfx_mod.pour(E["flow0"], E["flow1"], fl, shot_col, stream=0.35, tail_drips=0), E["flow0"], 0.07)
drip_times = list(np.arange(E["drip0"], E["flow0"] + 0.2, 0.21)) + list(np.arange(E["flow1"] - 0.1, E["flow1"] + 1.2, 0.26))
for td in drip_times:
    place(sfx, sfx_mod.bubble(rng.uniform(1300, 2500), 1.0, 0.4), td + rng.uniform(0, 0.04), 0.1, rng.choice([-0.12, 0.12]))
n = int((S["espresso"]["end"] - S["espresso"]["start"]) * SR)
boiler = (np.sin(2 * np.pi * 100 * tvec(n)) * 0.6 + np.sin(2 * np.pi * 200 * tvec(n)) * 0.25 + lp(rng.standard_normal(n), 400) * 0.6)
boiler *= smoothstep(0, 0.3, tvec(n)) * (1 - smoothstep(n / SR - 0.3, n / SR, tvec(n)))
place(sfx, boiler, S["espresso"]["start"], 0.02)

# espresso over the ice: thinner, darker stream and the ice crackling from the heat
p_rate = lambda t: flow_rate(t, P["t0"], P["t1"], 0.4)
place(sfx, sfx_mod.pour(P["t0"], P["t1"], p_rate, column, stream=0.6, tail_drips=6), P["t0"], 0.22, pour_pan)
place(sfx, sfx_mod.crackle(P["t1"] - P["t0"] + 3.5, lambda tt: 45 * np.exp(-tt / 2.0) + 3), P["t0"] + 0.25, 0.7, 0.05)

# stirring
ST = cues["stir"]
dur = ST["s1"] - ST["s0"] + 1.2
s_rate = lambda tt: smoothstep(0, 0.35, tt) * (1 - smoothstep(ST["s1"] - ST["s0"] - 0.3, ST["s1"] - ST["s0"] + 0.5, tt))
place(sfx, sfx_mod.stir_swirl(dur, s_rate, 0.85), ST["s0"], 0.13)
t_dip = ST["in0"] + (ST["in1"] - ST["in0"]) * 0.45  # straw breaks the surface
place(sfx, sfx_mod.bubble(520, 1.0, 0.8), t_dip, 0.12)
place(sfx, sfx_mod.noise_burst(60, 400, 5000), t_dip, 0.05)
tr = ST["s0"]
while tr < ST["s1"]:  # extra rattle the coarse contact list misses
    tr += rng.exponential(1 / 9)
    under = rng.random() < 0.6
    hit = sfx_mod.ice_hit(rng.uniform(20, 70), under) if rng.random() < 0.7 else sfx_mod.glass_hit(rng.uniform(15, 45), 0.8, under)
    place(sfx, hit, tr, 0.3 * float(s_rate(tr - ST["s0"])), rng.uniform(-0.3, 0.3))

# transitions
for s in tl["scenes"][1:]:
    place(sfx, sfx_mod.whoosh(1 if s["step"] % 2 else -1), s["start"] - 0.3, 0.4)

# UI: droplet pops for the step badge, ticks for the tag, sparkle for "완성"
place(sfx, sfx_mod.droplet_pop(420), S["hook"]["start"] + 0.06, 0.16)
for s in tl["scenes"]:
    if s["step"] > 0:
        place(sfx, sfx_mod.droplet_pop(rng.uniform(600, 720)), s["start"] + 0.08, 0.18, -0.3)
        place(sfx, sfx_mod.soft_tick(), s["start"] + 0.55, 0.1, -0.4)
place(sfx, sfx_mod.chime(1318.5), S["stir"]["voiceStart"] + S["stir"]["voiceDur"] - 0.6, 0.1)
place(sfx, sfx_mod.chime(1568.0), S["outro"]["start"] + 0.15, 0.13)

# ---------------------------------------------------------------- ambience

amb = buf()
pink = np.cumsum(rng.standard_normal(N)) * 0.002
pink = hp(pink - np.mean(pink), 60)
amb[:, 0] += lp(pink, 1800) * 0.25
amb[:, 1] += lp(np.roll(pink, 4000), 1800) * 0.25
walla = make_walla()
if walla is not None:
    amb += walla * 0.05
for t in np.arange(1.5, DUR - 1, 3.9):  # cups, saucers and spoons around the room
    kind = rng.random()
    hit = sfx_mod.glass_hit(rng.uniform(40, 90), 0.3, False) if kind < 0.5 else sfx_mod.straw_hit(rng.uniform(60, 120), False)
    place(amb, lp(hit, 3500), t + rng.uniform(0, 2), 0.05, rng.uniform(-0.85, 0.85))

# ---------------------------------------------------------------- music (lo-fi)

mus = buf()
BPM = 86
beat = 60 / BPM
bar = 4 * beat
chords = [  # Fmaj9, Em9, Dm9, Cmaj9
    [53, 57, 60, 64, 67], [52, 55, 59, 62, 66], [50, 53, 57, 60, 64], [48, 52, 55, 59, 62]]
roots = [41, 40, 38, 36]
mid = lambda m: 440 * 2 ** ((m - 69) / 12)
music_end = DUR - 0.4
nbars = int(np.ceil(music_end / bar))


def epiano(f, length, vel):
    t = tvec(int(length * SR))
    idx = 1.2 * np.exp(-t / 0.3) + 0.25
    mod = np.sin(2 * np.pi * f * t) * idx
    tone = np.sin(2 * np.pi * f * t + mod) + 0.18 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.4)
    env = np.minimum(1, t / 0.006) * np.exp(-t / 1.6)
    return tone * env * vel


def kick():
    t = tvec(int(0.4 * SR))
    f = 45 + 75 * np.exp(-t / 0.04)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22) + np.pad(sfx_mod.noise_burst(2, 1000, 6000), (0, len(t)))[:len(t)] * 0.3


def snare():
    t = tvec(int(0.3 * SR))
    noise = lp(bp(rng.standard_normal(len(t)), 900, 7000), 5000) * np.exp(-t / 0.11)
    return noise * 0.8 + np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.06) * 0.5


def hat(open_=False):
    t = tvec(int(0.2 * SR))
    return hp(rng.standard_normal(len(t)), 7000) * np.exp(-t / (0.09 if open_ else 0.03))


keys = np.zeros(N)
bass = np.zeros(N)
drums = buf()
for b in range(nbars):
    t0 = b * bar
    ci = b % 4
    last = b == nbars - 1
    for m in chords[ci]:  # chord on 1, lighter re-strike on the "and" of 2
        for off, vel, ln in ((0.0, 0.16, bar * (1.4 if last else 1.0)), (1.5 * beat, 0.08, 1.2)):
            if last and off > 0:
                continue
            sig = epiano(mid(m) * rng.uniform(0.998, 1.002), ln, vel * rng.uniform(0.85, 1.1))
            i = int((t0 + off + rng.uniform(0, 0.012)) * SR)
            keys[i:i + len(sig)] += sig[: max(0, N - i)]
    rt = tvec(int(bar * SR))
    for off, ln in ((0, 2.4 * beat), (2.5 * beat, 1.3 * beat)):
        n = int(ln * SR)
        tt = tvec(n)
        f = mid(roots[ci])
        bline = (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)) * np.minimum(1, tt / 0.01) * np.exp(-tt / 0.9)
        i = int((t0 + off) * SR)
        bass[i:i + n] += bline[: max(0, N - i)] * 0.3
    if b == 0 or last:
        continue
    for k in range(4):
        tb = t0 + k * beat
        if k in (0, 2):
            place(drums, kick(), tb, 0.55)
        if k in (1, 3):
            place(drums, snare(), tb + 0.012, 0.22, 0.05)
        for e8 in (0, 1):
            place(drums, hat(open_=(k == 3 and e8 == 1)), tb + e8 * beat * 0.58, 0.05 if e8 else 0.08, 0.25)
    if b % 2 == 1:
        place(drums, kick(), t0 + 2.75 * beat, 0.3)

trem = 1 + 0.12 * np.sin(2 * np.pi * 4.2 * tvec(N))
mus[:, 0] += keys * trem + bass
mus[:, 1] += keys * (2 - trem) + bass
mus += drums
crackle = np.zeros(N)
idx = rng.choice(N, int(DUR * 7), replace=False)
crackle[idx] = rng.uniform(-1, 1, len(idx)) * 0.25
mus += hp(lp(crackle, 6000), 800)[:, None] * 0.15 + lp(rng.standard_normal((N, 2)), 4000) * 0.0025
mus = lp(np.tanh(mus * 1.4) / 1.4, 6500)
tt = tvec(N)
mus *= (smoothstep(0, 0.6, tt) * (1 - smoothstep(music_end - 2.2, music_end, tt)))[:, None]

# ---------------------------------------------------------------- voice

voice = np.zeros(N)
for s in tl["scenes"]:
    x, sr = sf.read(os.path.join(B, "voice", f"{s['id']}.wav"))
    i = int(s["voiceStart"] * SR)
    voice[i:i + len(x)] += x[: max(0, N - i)]

# ---------------------------------------------------------------- spaces + mix

def reverb(x, ir, wet):
    y = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:N] for c in range(2)], 1)
    return x + y * wet


small = sfx_mod.room_ir(0.55, 7000)
sfx = reverb(sfx, small, 0.16)
mus = reverb(mus, sfx_mod.room_ir(1.1, 5000, 1.6, early=False), 0.16)
voice_st = np.stack([voice, voice], 1)
voice_st = reverb(voice_st, small, 0.04)

# duck the music (and a touch of the ambience) under the narration
venv = np.sqrt(signal.lfilter([0.0015], [1, -0.9985], voice ** 2))
active = np.clip(venv / 0.05, 0, 1)
duck = signal.lfilter([0.0006], [1, -0.9994], active)
duck = np.clip(duck * 2.2, 0, 1)
music_gain = 0.42 * (1 - 0.55 * duck)

sfx_gain = 0.9 * (1 - 0.3 * duck)
mix = voice_st * 1.0 + sfx * sfx_gain[:, None] + amb * 0.9 + mus * music_gain[:, None]
peak = np.max(np.abs(mix))
mix = np.tanh(mix / max(peak, 1e-6) * 1.15) * 0.93

os.makedirs(OUT, exist_ok=True)
sf.write(os.path.join(OUT, "mix.wav"), mix.astype(np.float32), SR, subtype="PCM_24")
for name, x in (("voice", voice_st), ("sfx", sfx), ("ambience", amb), ("music", mus)):
    sf.write(os.path.join(OUT, f"stem_{name}.wav"), (x / (np.max(np.abs(x)) + 1e-9) * 0.9).astype(np.float32), SR, subtype="PCM_16")
print(f"mix: {DUR:.2f}s, {len(ev['events'])} physics contacts voiced")

# level report per scene (dBFS RMS of each component as it sits in the mix)
scale = 1.15 / max(peak, 1e-6)
db = lambda x: 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
print(f"{'scene':9s} {'voice':>7s} {'sfx':>7s} {'amb':>7s} {'music':>7s}")
for s in tl["scenes"]:
    a, b = int(s["start"] * SR), int(s["end"] * SR)
    va, vb = int(s["voiceStart"] * SR), int((s["voiceStart"] + s["voiceDur"]) * SR)
    print(f"{s['id']:9s} {db(voice_st[va:vb] * scale):7.1f} {db(sfx[a:b] * 0.9 * scale):7.1f} {db(amb[a:b] * 0.9 * scale):7.1f} {db((mus * music_gain[:, None])[va:vb] * scale):7.1f}")
