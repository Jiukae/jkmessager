// Small deterministic noise helpers used for procedural textures and motion.

export function mulberry32(seed) {
	let a = seed >>> 0;
	return function () {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function hash2(ix, iy, seed) {
	let h = (ix * 374761393 + iy * 668265263 + seed * 2147483647) | 0;
	h = Math.imul(h ^ (h >>> 13), 1274126177);
	h ^= h >>> 16;
	return (h >>> 0) / 4294967296;
}

const fade = (t) => t * t * (3 - 2 * t);

// periodic value noise: wraps every `px` x `py` cells so textures tile
export function vnoise(x, y, seed = 0, px = 1 << 20, py = 1 << 20) {
	const ix = Math.floor(x), iy = Math.floor(y);
	const fx = fade(x - ix), fy = fade(y - iy);
	const x0 = ((ix % px) + px) % px, x1 = (((ix + 1) % px) + px) % px;
	const y0 = ((iy % py) + py) % py, y1 = (((iy + 1) % py) + py) % py;
	const a = hash2(x0, y0, seed), b = hash2(x1, y0, seed);
	const c = hash2(x0, y1, seed), d = hash2(x1, y1, seed);
	return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
}

export function fbm(x, y, octaves = 5, seed = 0, px = 1 << 20, py = 1 << 20) {
	let sum = 0, amp = 0.5, norm = 0;
	for (let o = 0; o < octaves; o++) {
		sum += amp * vnoise(x, y, seed + o * 17, px, py);
		norm += amp;
		x *= 2; y *= 2; px *= 2; py *= 2; amp *= 0.5;
	}
	return sum / norm;
}

// 1D smooth noise for camera shake etc.
export function noise1(t, seed = 0) {
	const i = Math.floor(t);
	const f = fade(t - i);
	const a = hash2(i, 0, seed) * 2 - 1;
	const b = hash2(i + 1, 0, seed) * 2 - 1;
	return a + (b - a) * f;
}

export const clamp01 = (x) => Math.min(1, Math.max(0, x));
export const smooth = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };
export const easeInOut = (t) => { t = clamp01(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
export const easeOut = (t) => 1 - Math.pow(1 - clamp01(t), 3);
export const lerp = (a, b, t) => a + (b - a) * t;
