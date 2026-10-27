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
    return signal.sosfilt(signal.butter(order, [lo, hi], "bandpass", fs=SR, output="sos"), x)


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


def visible_main(t):
    """1 while the glass is on screen (the espresso scene cuts to the machine)."""
    e = S["espresso"]
    return 0.0 if e["start"] + 0.05 < t < e["end"] - 0.05 else 1.0


# ---------------------------------------------------------------- sources

def damped(freqs, decays, amps, length, phase=True):
    t = tvec(int(length * SR))
    out = np.zeros_like(t)
    for f, d, a in zip(freqs, decays, amps):
        ph = rng.uniform(0, 2 * np.pi) if phase else 0
        out += a * np.sin(2 * np.pi * f * t + ph) * np.exp(-t / d)
    return out


def click(length=0.004, lo=2000, hi=12000):
    n = int(length * SR)
    x = rng.standard_normal(n) * np.exp(-np.arange(n) / (n / 4))
    return bp(np.concatenate([x, np.zeros(256)]), lo, min(hi, SR / 2 - 100))


def glass_clink(v, fill, under):
    """Ice hitting the tumbler: inharmonic shell modes, lower + deader when full."""
    amp = min(1.0, (v / 160) ** 0.85)
    f0 = 1240 * (1 - 0.32 * fill) * rng.uniform(0.97, 1.03)
    damp = 1 - 0.6 * fill
    s = damped([f0, f0 * 2.71, f0 * 5.13, f0 * 8.3], [0.42 * damp, 0.2 * damp, 0.09 * damp, 0.04 * damp], [1, 0.6, 0.32, 0.14], 1.0)
    s[: 8] *= np.linspace(0, 1, 8)
    tick = damped(rng.uniform(2800, 5600, 2), [0.012, 0.02], [0.5, 0.35], 0.06)
    s[: len(tick)] += tick
    c = click(0.003)
    s[: len(c)] += c * 0.6
    if under:
        s = lp(s, 1300) * 0.35
    return s * amp


def ice_tick(v, under, plastic=False):
    amp = min(1.0, (v / 130) ** 0.85)
    if plastic:
        s = damped(rng.uniform(1100, 2300, 2), [0.018, 0.012], [0.6, 0.4], 0.08)
    else:
        s = damped(rng.uniform(2300, 5600, 3), rng.uniform(0.012, 0.035, 3), [0.6, 0.45, 0.3], 0.12)
        c = click(0.002, 3000)
        s[: len(c)] += c * 0.5
    if under:
        s = lp(s, 900) * 0.4
    return s * amp


def bubble(f, amp, rising=0.12):
    d = 0.043 * f + 0.0014 * f ** 1.5
    n = int(min(5.5 / d, 0.12) * SR)
    t = tvec(n)
    inst = f * (1 + rising * d * t)
    return amp * np.sin(2 * np.pi * np.cumsum(inst) / SR) * np.exp(-d * t)


def bubbles(bus, t0, t1, rate_fn, per_sec, fmin, fmax, gain, pan=0.0):
    """Liquid noise as a cloud of Minnaert bubbles, density following rate_fn(t)."""
    t = t0
    while t < t1:
        t += rng.exponential(1 / per_sec)
        r = float(rate_fn(t))
        if rng.random() > r:
            continue
        f = np.exp(rng.uniform(np.log(fmin), np.log(fmax)))
        a = (fmin / f) ** 0.6 * rng.uniform(0.3, 1.0)
        place(bus, bubble(f, a), t, gain, pan + rng.uniform(-0.15, 0.15))


def resonant_noise(t0, t1, rate_fn, freq_fn, q, lo=300, hi=9000):
    """Pour noise through an air-column resonance whose pitch rises as the glass fills."""
    n = int((t1 - t0) * SR)
    t = t0 + tvec(n)
    x = bp(rng.standard_normal(n), lo, hi)
    flutter = 1 + 0.5 * bp(rng.standard_normal(n), 6, 30, 1) * 8
    x *= np.clip(flutter, 0.2, 2.0) * rate_fn(t)
    out = np.zeros(n)
    blk = 256
    zi = np.zeros(2)
    for i in range(0, n, blk):
        f = float(freq_fn(t[i]))
        b, a = signal.iirpeak(min(f, 8000), q, fs=SR)
        out[i:i + blk], zi = signal.lfilter(b, a, x[i:i + blk], zi=zi)
    return out, x


def air_column(lv):
    """Quarter-wave resonance of the air above the liquid (with end correction)."""
    L = (H - lv + 2.3) / 100
    return 343 / (4 * L)


# ---------------------------------------------------------------- SFX bus

sfx = buf()

# physics contacts -> clinks
for e in ev["events"]:
    t = e["t"]
    vis = visible_main(t)
    if vis == 0:
        continue
    fill = float(np.clip((e["level"] - BASE) / (H - BASE), 0, 1)) if e["level"] > 0 else 0.0
    pan = float(np.clip(rng.normal(0, 0.18), -0.5, 0.5))
    if e["kind"] == "glass":
        place(sfx, glass_clink(e["v"], fill, e["under"]), t, 0.5, pan)
    elif e["kind"] == "straw":
        place(sfx, ice_tick(e["v"], e["under"], plastic=True), t, 0.35, pan)
    else:
        place(sfx, ice_tick(e["v"], e["under"]), t, 0.32, pan)

# hero shots: a few lazy clinks of settling ice
for t, v in ((0.75, 45), (2.3, 30), (3.85, 38), (S["outro"]["start"] + 0.9, 40), (S["outro"]["start"] + 2.6, 28)):
    place(sfx, glass_clink(v, 0.85, False), t, 0.38, rng.uniform(-0.2, 0.2))
    place(sfx, ice_tick(v * 0.8, False), t + 0.05, 0.2)

# water pour: bubbles + rising air-column resonance + splash on the ice
w_rate = lambda t: flow_rate(t, W["t0"], W["t1"], 0.35)
bubbles(sfx, W["t0"], W["t1"] + 0.3, w_rate, 1400, 650, 4800, 0.11, 0.15)
res, raw = resonant_noise(W["t0"], W["t1"] + 0.3, w_rate, lambda t: air_column(float(level(t))), 6)
place(sfx, res * 0.09 + hp(raw, 2500) * 0.02, W["t0"], 1.0, 0.1)
for _ in range(14):  # bobbing ice nudging the glass while the water churns
    t = rng.uniform(W["t0"] + 0.5, W["t1"])
    place(sfx, glass_clink(rng.uniform(15, 35), float((level(t) - BASE) / (H - BASE)), True), t, 0.4)

# espresso machine
E = cues["espresso"]
e0, e1 = E["pumpOn"], E["pumpOff"]
n = int((e1 - e0 + 0.15) * SR)
t = tvec(n)
fpump = 50 - 3 * smoothstep(0, 1.6, t)
ph = 2 * np.pi * np.cumsum(fpump) / SR
pulse = np.maximum(np.sin(ph), 0) ** 3
pump = lp(pulse - pulse.mean(), 1800, 4) * 1.0 + lp(rng.standard_normal(n) * pulse, 3000) * 0.18
pump += 0.4 * np.sin(ph) * smoothstep(0, 0.4, t)
pump *= smoothstep(0, 0.05, t) * (1 - smoothstep(len(t) / SR - 0.2, len(t) / SR, t)) * (0.7 + 0.3 * smoothstep(0.3, 1.8, t))
place(sfx, pump, e0, 0.22, -0.1)
for tc in (e0, e1):
    place(sfx, damped([180, 1450, 3200], [0.025, 0.012, 0.006], [1, 0.5, 0.3], 0.1) + np.pad(click(0.003, 1500), (0, 4800))[:4800], tc, 0.35)
hiss = bp(rng.standard_normal(int(0.6 * SR)), 1500, 7000) * np.exp(-tvec(int(0.6 * SR)) / 0.16)
place(sfx, hiss, e1 + 0.03, 0.08, 0.15)
fl = lambda t: flow_rate(t, E["flow0"], E["flow1"], 0.5)
bubbles(sfx, E["flow0"], E["flow1"] + 0.2, fl, 700, 1800, 6500, 0.045)
drip_times = list(np.arange(E["drip0"], E["flow0"] + 0.2, 0.21)) + list(np.arange(E["flow1"] - 0.1, E["flow1"] + 1.2, 0.26))
for td in drip_times:
    place(sfx, bubble(rng.uniform(1300, 2500), 1.0, 0.3), td + rng.uniform(0, 0.04), 0.1, rng.choice([-0.12, 0.12]))

# espresso poured over the ice: thinner stream, rising pitch, thermal crackle of the ice
p_rate = lambda t: flow_rate(t, P["t0"], P["t1"], 0.4)
bubbles(sfx, P["t0"], P["t1"] + 0.3, p_rate, 900, 900, 5200, 0.12, -0.1)
res, raw = resonant_noise(P["t0"], P["t1"] + 0.3, p_rate, lambda t: air_column(float(level(t))), 7)
place(sfx, res * 0.08 + hp(raw, 3000) * 0.012, P["t0"], 1.0, -0.05)
tc = P["t0"] + 0.25
while tc < P["t1"] + 3.0:
    tc += rng.exponential(1 / (32 * np.exp(-(tc - P["t0"]) / 2.2) + 2))
    place(sfx, damped(rng.uniform(3000, 7500, 2), [0.004, 0.006], [1, 0.6], 0.03), tc, rng.uniform(0.02, 0.09), rng.uniform(-0.3, 0.3))

# stirring swish
ST = cues["stir"]
n = int((ST["s1"] - ST["s0"] + 1.0) * SR)
t = tvec(n)
rate = smoothstep(0, 0.35, t) * (1 - smoothstep(ST["s1"] - ST["s0"] - 0.3, ST["s1"] - ST["s0"] + 0.4, t))
swish = bp(rng.standard_normal(n), 250, 1700) * (0.35 + 0.65 * np.abs(np.sin(np.pi * 0.85 * 2 * t))) * rate
place(sfx, swish, ST["s0"], 0.08)
place(sfx, ice_tick(60, False, plastic=True), ST["in1"] - 0.05, 0.3)

# whoosh on every cut
for i, s in enumerate(tl["scenes"][1:]):
    n = int(0.55 * SR)
    t = tvec(n)
    x = rng.standard_normal(n)
    env = np.exp(-((t - 0.3) / 0.11) ** 2)
    out = np.zeros(n)
    zi = np.zeros(4)
    for k in range(0, n, 256):
        fc = 300 + 2600 * np.exp(-((t[k] - 0.3) / 0.12) ** 2)
        sos = signal.butter(2, [fc * 0.6, min(fc * 1.6, 20000)], "bandpass", fs=SR, output="sos")
        if k == 0:
            zi = signal.sosfilt_zi(sos) * 0
        out[k:k + 256], zi = signal.sosfilt(sos, x[k:k + 256], zi=zi)
    out *= env
    d = 1 if (s["step"] % 2) else -1
    pan = np.clip((t - 0.3) * 4 * d, -0.8, 0.8)
    st = np.stack([out * np.cos((pan + 1) * np.pi / 4), out * np.sin((pan + 1) * np.pi / 4)], 1)
    place(sfx, st, s["start"] - 0.3, 0.55)

# UI: pops for the step badge/tag, sparkle for "완성"
def pop(f0=520, f1=1300, length=0.07):
    t = tvec(int(length * SR))
    f = f0 + (f1 - f0) * (t / length)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.025)


def chime(base):
    out = np.zeros(int(1.6 * SR))
    for k, ratio in enumerate((1, 1.26, 1.5, 2.0)):
        f = base * ratio
        tone = damped([f, f * 2.76], [0.6, 0.25], [1, 0.25], 1.2)
        i = int(k * 0.065 * SR)
        out[i:i + len(tone)] += tone * (0.9 - 0.12 * k)
    return out


place(sfx, pop(380, 900, 0.09), S["hook"]["start"] + 0.06, 0.16)
for s in tl["scenes"]:
    if s["step"] > 0:
        place(sfx, pop(), s["start"] + 0.08, 0.13, -0.3)
        place(sfx, pop(1500, 2200, 0.035), s["start"] + 0.55, 0.08, -0.4)
place(sfx, chime(1318.5), S["stir"]["voiceStart"] + S["stir"]["voiceDur"] - 0.6, 0.11)
place(sfx, chime(1568.0), S["outro"]["start"] + 0.15, 0.14)

# ---------------------------------------------------------------- ambience

amb = buf()
pink = np.cumsum(rng.standard_normal(N)) * 0.002
pink = hp(pink - np.mean(pink), 60)
amb[:, 0] += lp(pink, 1800) * 0.25
amb[:, 1] += lp(np.roll(pink, 4000), 1800) * 0.25
babble = np.zeros(N)
for _ in range(7):
    am = np.clip(lp(rng.standard_normal(N), 5, 1) * 40 + 0.2, 0, 1)
    babble += bp(rng.standard_normal(N), rng.uniform(250, 500), rng.uniform(1200, 2600)) * am
amb[:, 0] += babble * 0.012
amb[:, 1] += np.roll(babble, 9000) * 0.012
for t in np.arange(1.5, DUR - 1, 4.7):  # far away cups
    place(amb, lp(glass_clink(rng.uniform(40, 80), 0.6, False), 3000), t + rng.uniform(0, 2), 0.05, rng.uniform(-0.8, 0.8))

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
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22) + np.pad(click(0.002, 1000, 6000), (0, len(t)))[:len(t)] * 0.3


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

def room_ir(length, decay, bright):
    t = tvec(int(length * SR))
    ir = rng.standard_normal((len(t), 2)) * np.exp(-t / decay)[:, None]
    ir = lp(ir, bright)
    ir[0] = 0
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


def reverb(x, ir, wet):
    y = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:N] for c in range(2)], 1)
    return x + y * wet


small = room_ir(0.9, 0.18, 6000)
sfx = reverb(sfx, small, 0.22)
mus = reverb(mus, room_ir(1.6, 0.35, 5000), 0.18)
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
