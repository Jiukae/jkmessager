"""Physically-motivated sound models for the reel (all synthesized, mono unless noted).

Impacts use modal synthesis: a short contact-force pulse (its width sets how bright
the hit is) drives a bank of damped resonances. Glass shell modes come in slightly
detuned pairs, which gives the slow shimmer of a real tumbler. Liquids are built
from Minnaert bubbles plus stream turbulence, coloured by the air column above the
liquid, which rises in pitch as the vessel fills.
"""
import numpy as np
from scipy import signal

SR = 48000
rng = np.random.default_rng(1234)


def seed(s):
    global rng
    rng = np.random.default_rng(s)


def tvec(n):
    return np.arange(n) / SR


def bp(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, min(hi, SR / 2 - 200)], "bandpass", fs=SR, output="sos"), x, axis=0)


def lp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "lowpass", fs=SR, output="sos"), x, axis=0)


def hp(x, f, order=2):
    return signal.sosfilt(signal.butter(order, f, "highpass", fs=SR, output="sos"), x, axis=0)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def modal(freqs, decays, amps, length, beat=0.0):
    """Damped resonances; with beat > 0 each mode is a detuned pair (split shell modes)."""
    t = tvec(int(length * SR))
    out = np.zeros_like(t)
    for f, d, a in zip(freqs, decays, amps):
        if f >= SR / 2 - 500:
            continue
        ph = rng.uniform(0, 2 * np.pi)
        if beat > 0:
            db = beat * rng.uniform(0.5, 1.5)
            out += a * 0.5 * (np.sin(2 * np.pi * (f - db / 2) * t + ph) + np.sin(2 * np.pi * (f + db / 2) * t + ph * 1.3)) * np.exp(-t / d)
        else:
            out += a * np.sin(2 * np.pi * f * t + ph) * np.exp(-t / d)
    return out


def strike(ir, width_ms):
    """Excite a resonator with a half-sine contact pulse: shorter = harder = brighter."""
    n = max(2, int(width_ms * 1e-3 * SR))
    pulse = np.sin(np.pi * np.arange(n) / n)
    return np.convolve(ir, pulse / pulse.sum())[: len(ir)]


def noise_burst(ms, lo, hi, decay_frac=0.25):
    n = int(ms * 1e-3 * SR)
    x = rng.standard_normal(n) * np.exp(-np.arange(n) / max(1, n * decay_frac))
    return bp(np.concatenate([x, np.zeros(512)]), lo, hi)


def norm(x, q=99.9):
    """Scale so the q-th percentile of |x| is 1 (robust peak normalisation)."""
    return x / (np.percentile(np.abs(x), q) + 1e-12)


def mix_into(a, b, gain=1.0):
    n = min(len(a), len(b))
    a[:n] += b[:n] * gain
    return a


# ---------------------------------------------------------------- impacts

def glass_hit(v, fill, under, low=False):
    """Ice striking the tumbler. fill 0..1 lowers and deadens the ring; low=True adds the thick base."""
    amp = min(1.0, (v / 170) ** 0.9)
    f0 = 1180 * (1 - 0.34 * fill ** 0.8) * rng.uniform(0.97, 1.03)
    damp = 1 - 0.72 * fill
    ratios = [1, 2.74, 5.17, 8.35, 12.2]
    decays = np.array([1.0, 0.5, 0.22, 0.1, 0.05]) * damp
    amps = np.array([1, 0.7, 0.45, 0.25, 0.12]) * rng.uniform(0.6, 1.3, 5)
    ir = modal([f0 * r for r in ratios], decays, amps, 1.4, beat=1.1)
    if low:  # foot of the glass: a dull thunk under the ring
        ir += modal([rng.uniform(380, 460), rng.uniform(900, 1100)], [0.035, 0.02], [1.6, 0.6], 1.4)
    s = strike(ir, 0.18 + 0.35 * (1 - amp))
    mix_into(s, modal(rng.uniform(4200, 9000, 3), rng.uniform(0.004, 0.012, 3), [0.6, 0.4, 0.3], 0.05))  # the cube itself
    mix_into(s, noise_burst(2.0, 2500, 14000), 0.5)
    if under:
        s = lp(s, 1100) * 0.32
        mix_into(s, bubble(rng.uniform(500, 900), 0.25, 0.5))
    return s * amp


def ice_hit(v, under):
    """Two cubes knocking: a hard, short, high clack."""
    amp = min(1.0, (v / 140) ** 0.9)
    ir = modal(rng.uniform(2300, 7200, 4), rng.uniform(0.006, 0.03, 4), rng.uniform(0.3, 1.0, 4), 0.12)
    s = strike(ir, 0.12 + 0.2 * (1 - amp))
    mix_into(s, noise_burst(1.5, 3000, 16000), 0.6)
    if under:
        s = lp(s, 700) * 0.45
    return s * amp


def straw_hit(v, under):
    """Hollow plastic straw: low tube modes, quickly damped."""
    amp = min(1.0, (v / 130) ** 0.9)
    f = rng.uniform(700, 1000)
    s = strike(modal([f, f * 2.8, f * 5.2], [0.03, 0.015, 0.008], [1, 0.5, 0.25], 0.12), 0.5)
    mix_into(s, noise_burst(2.5, 1500, 8000), 0.3)
    if under:
        s = lp(s, 800) * 0.5
    return s * amp


# ---------------------------------------------------------------- liquids

def bubble(f, amp, rising=0.12):
    d = 0.043 * f + 0.0014 * f ** 1.5
    n = int(min(5.5 / d, 0.15) * SR)
    t = tvec(n)
    inst = f * (1 + rising * d * t)
    return amp * np.sin(2 * np.pi * np.cumsum(inst) / SR) * np.exp(-d * t) * np.minimum(1, t / 0.0006)


def bubble_cloud(n, rate_fn, per_sec, fmin, fmax, t0=0.0):
    """Mono buffer of n samples; bubble density follows rate_fn(absolute time)."""
    out = np.zeros(n)
    t = 0.0
    while t < n / SR:
        t += rng.exponential(1 / per_sec)
        if rng.random() > float(rate_fn(t0 + t)):
            continue
        f = np.exp(rng.uniform(np.log(fmin), np.log(fmax)))
        b = bubble(f, (fmin / f) ** 0.55 * rng.lognormal(0, 0.45))
        i = int(t * SR)
        out[i:i + len(b)] += b[: max(0, n - i)]
    return out


def resonate(x, t0, freq_fns, qs, gains):
    """Time-varying peak filters (one per air-column harmonic)."""
    out = np.zeros_like(x)
    blk = 256
    for fn, q, g in zip(freq_fns, qs, gains):
        zi = np.zeros(2)
        y = np.zeros_like(x)
        for i in range(0, len(x), blk):
            f = min(float(fn(t0 + i / SR)), 15000)
            b, a = signal.iirpeak(f, q, fs=SR)
            y[i:i + blk], zi = signal.lfilter(b, a, x[i:i + blk], zi=zi)
        out += y * g
    return out


def pour(t0, t1, rate_fn, column_fn, stream=1.0, tail_drips=8):
    """A stream into a glass: turbulence + bubbles, coloured by the rising air column,
    a splash on contact and a few trailing drips. Returns a mono buffer starting at t0."""
    n = int((t1 - t0 + 1.8) * SR)
    t = t0 + tvec(n)
    rate = np.asarray(rate_fn(t), dtype=float)
    turb_env = np.abs(lp(rng.standard_normal(n), 45, 2)) * 9
    turb = bp(rng.standard_normal(n), 180, 7500) * np.clip(turb_env, 0, 2.5) * rate
    cloud = bubble_cloud(n, rate_fn, 1500 * stream, 500 / stream ** 0.3, 5200, t0)
    dry = turb * 0.55 + cloud
    f1 = column_fn
    wet = resonate(dry, t0, [f1, lambda tt: 3 * f1(tt)], [9, 7], [1.0, 0.45])
    out = dry * 0.55 + wet * 1.1
    # splash on first contact
    on = np.argmax(rate > 0.05)
    sp = bp(rng.standard_normal(int(0.25 * SR)), 300, 9000) * np.exp(-tvec(int(0.25 * SR)) / 0.06)
    out[on:on + len(sp)] += sp[: max(0, n - on)] * 0.9 * stream
    # last drips after the stream breaks
    td = t1 - t0 + 0.05
    gap = 0.11
    for k in range(tail_drips):
        td += gap * rng.uniform(0.8, 1.3)
        gap *= 1.35
        i = int(td * SR)
        if i >= n:
            break
        d = bubble(rng.uniform(800, 1700), 0.7 * (0.85 ** k), 0.6)
        out[i:i + len(d)] += d[: max(0, n - i)]
        sp = noise_burst(4, 1500, 9000)
        out[i:i + len(sp)] += sp[: max(0, n - i)] * 0.15 * (0.85 ** k)
    return norm(out)


def crackle(dur, density_fn, loud=1.0):
    """Ice fracturing under warmer liquid: clustered sharp cracks and occasional pops."""
    n = int(dur * SR)
    out = np.zeros(n)
    t = 0.0
    while t < dur:
        t += rng.exponential(1 / max(density_fn(t), 0.5))
        burst = 1 if rng.random() > 0.15 else rng.integers(3, 7)
        tb = t
        for _ in range(burst):
            tb += rng.exponential(0.008)
            big = rng.random() < 0.08
            f = rng.uniform(1500, 3000, 2) if big else rng.uniform(3000, 9500, 2)
            c = strike(modal(f, rng.uniform(0.002, 0.009 if not big else 0.02, 2), [1, 0.6], 0.05), 0.05)
            mix_into(c, noise_burst(0.6, 2000, 16000), 0.8)
            a = rng.lognormal(-1.2, 0.6) * (2.5 if big else 1.0) * loud
            i = int(tb * SR)
            out[i:i + len(c)] += c[: max(0, n - i)] * a
    return out / (np.max(np.abs(out)) + 1e-12) * loud


# ---------------------------------------------------------------- espresso machine

def rocker_click():
    s = strike(modal([2400, 4100, 6300], [0.006, 0.004, 0.003], [1, 0.7, 0.4], 0.06), 0.1)
    s2 = strike(modal([1800, 3300], [0.008, 0.005], [0.8, 0.5], 0.06), 0.15)
    out = np.zeros(int(0.12 * SR))
    mix_into(out, s)
    mix_into(out[int(0.045 * SR):], s2, 0.7)
    return out


def solenoid(release=False):
    """Valve clunk; on release the trapped pressure dumps into the drip tray."""
    body = strike(modal([160, 420, 1150, 2600], [0.05, 0.03, 0.012, 0.006], [1, 0.7, 0.4, 0.25], 0.3), 0.6)
    if not release:
        return norm(body)
    n = int(0.7 * SR)
    t = tvec(n)
    hiss = bp(rng.standard_normal(n), 900, 7000) * np.exp(-t / 0.09) * 0.9
    dump = lp(rng.standard_normal(n), 900) * np.exp(-((t - 0.12) / 0.12) ** 2) * 0.8
    gurgle = bubble_cloud(n, lambda tt: float(np.exp(-((tt - 0.15) / 0.1) ** 2)), 600, 300, 1500)
    out = norm(hiss) * 0.6 + norm(dump) * 0.5 + norm(gurgle) * 0.35
    mix_into(out, norm(body), 0.8)
    return out


def vibratory_pump(dur, hz=50.0):
    """ULKA-style pump: 50 Hz piston strokes ringing the machine body, plus valve rattle.
    Strokes get heavier and slightly slower as brew pressure builds."""
    n = int(dur * SR)
    t = tvec(n)
    load = smoothstep(0.0, 2.2, t)
    f = hz * (1.04 - 0.08 * load)
    ph = np.cumsum(f) / SR
    strokes = np.where(np.diff(np.floor(ph), prepend=0) > 0)[0]
    imp = np.zeros(n)
    imp[strokes] = (0.6 + 0.4 * load[strokes]) * rng.uniform(0.85, 1.15, len(strokes))
    body = modal([96, 205, 370, 740, 1460, 2900], [0.09, 0.06, 0.04, 0.022, 0.012, 0.006], [1, 0.8, 0.6, 0.45, 0.3, 0.15], 0.2)
    out = signal.fftconvolve(imp, strike(body, 1.2))[:n]
    rattle = lp(hp(rng.standard_normal(n), 1800), 8000, 4) * np.convolve(imp, np.exp(-np.arange(200) / 40))[:n]
    hum = np.sin(2 * np.pi * np.cumsum(2 * f) / SR) * 0.25
    out = norm(out) + norm(rattle) * 0.15 + hum * 0.3
    env = smoothstep(0, 0.06, t) * (1 - smoothstep(dur - 0.08, dur, t))
    return norm(out * env)


# ---------------------------------------------------------------- stirring, UI, transitions

def stir_swirl(dur, rate_fn, rev_hz):
    """Liquid sloshing round the glass: two swells per turn, brightness following the speed."""
    n = int(dur * SR)
    t = tvec(n)
    r = np.asarray(rate_fn(t), dtype=float)
    lobes = np.abs(np.sin(np.pi * rev_hz * 2 * t)) ** 1.5
    x = bp(rng.standard_normal(n), 140, 1400)
    x = resonate(x, 0, [lambda tt: 420 + 260 * np.sin(2 * np.pi * rev_hz * tt)], [3], [1.0]) * 0.8 + x * 0.4
    return norm(x * (0.3 + 0.7 * lobes) * r)


def whoosh(direction=1.0):
    """Stereo transition: airy sweep + sub thump at the cut + a tiny tick. Cut at 0.3 s."""
    n = int(0.6 * SR)
    t = tvec(n)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    zi = None
    for k in range(0, n, 256):
        fc = 280 + 3200 * np.exp(-((t[k] - 0.3) / 0.1) ** 2)
        sos = signal.butter(2, [fc * 0.55, min(fc * 1.7, 20000)], "bandpass", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        out[k:k + 256], zi = signal.sosfilt(sos, x[k:k + 256], zi=zi)
    out *= np.exp(-((t - 0.3) / 0.1) ** 2)
    sub = np.zeros(n)
    i = int(0.29 * SR)
    ts = tvec(n - i)
    sub[i:] = np.sin(2 * np.pi * np.cumsum(38 + 50 * np.exp(-ts / 0.05)) / SR) * np.exp(-ts / 0.12) * 0.7
    pan = np.clip((t - 0.3) * 4.5 * direction, -0.85, 0.85)
    a = (pan + 1) * np.pi / 4
    st = np.stack([out * np.cos(a), out * np.sin(a)], 1) + sub[:, None] * 0.7
    tick = noise_burst(1.2, 3000, 12000) * 0.3
    st[int(0.3 * SR): int(0.3 * SR) + len(tick)] += tick[:, None]
    return st


def droplet_pop(f0=650):
    """UI pop styled as a water droplet."""
    return bubble(f0, 1.0, rising=1.6)[: int(0.12 * SR)]


def soft_tick():
    return strike(modal([2100, 3900], [0.012, 0.006], [1, 0.4], 0.05), 0.2)


def chime(base):
    out = np.zeros(int(1.8 * SR))
    for k, ratio in enumerate((1, 1.26, 1.5, 2.0)):
        f = base * ratio
        tone = modal([f, f * 2.76], [0.7, 0.25], [1, 0.22], 1.4, beat=0.8)
        i = int(k * 0.065 * SR)
        out[i:i + len(tone)] += tone * (0.9 - 0.12 * k)
    return out


# ---------------------------------------------------------------- space

def room_ir(rt60=0.55, bright=7000, length=1.2, early=True):
    """Small cafe: early reflections + diffuse tail whose highs die faster than its lows."""
    n = int(length * SR)
    t = tvec(n)
    ir = np.zeros((n, 2))
    bands = ((0, 500, 1.0), (500, 2500, 0.8), (2500, bright, 0.55))
    for lo, hi, k in bands:
        noise = rng.standard_normal((n, 2))
        noise = lp(noise, hi) if lo == 0 else bp(noise, lo, hi)
        ir += noise * np.exp(-6.9 * t / (rt60 * k))[:, None]
    ir *= smoothstep(0.0, 0.012, t)[:, None]
    if early:
        for d_ms, g in ((7, 0.6), (11, 0.5), (17, 0.42), (23, 0.35), (31, 0.3), (43, 0.22)):
            for c in range(2):
                i = int((d_ms + rng.uniform(-1.5, 1.5)) * 1e-3 * SR)
                ir[i, c] += g * rng.choice([-1, 1]) * 6
    ir[0] = 0
    return ir / np.sqrt(np.sum(ir ** 2) / 2)
