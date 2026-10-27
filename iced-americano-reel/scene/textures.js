// Procedural textures (no image downloads): wood table, blurred cafe backdrop,
// condensation / ice normal maps, contact shadow and caustic decals.
import * as THREE from 'three';
import { fbm, vnoise, mulberry32 } from './noise.js';

function canvas(w, h) {
	const c = document.createElement('canvas');
	c.width = w; c.height = h;
	return c;
}

function heightToNormal(height, w, h, strength, wrapX = true, wrapY = true) {
	const data = new Uint8ClampedArray(w * h * 4);
	const at = (x, y) => {
		x = wrapX ? (x + w) % w : Math.min(w - 1, Math.max(0, x));
		y = wrapY ? (y + h) % h : Math.min(h - 1, Math.max(0, y));
		return height[y * w + x];
	};
	for (let y = 0; y < h; y++) {
		for (let x = 0; x < w; x++) {
			const dx = (at(x + 1, y) - at(x - 1, y)) * strength;
			const dy = (at(x, y + 1) - at(x, y - 1)) * strength;
			let nx = -dx, ny = dy, nz = 1;
			const l = Math.hypot(nx, ny, nz);
			nx /= l; ny /= l; nz /= l;
			const i = (y * w + x) * 4;
			data[i] = (nx * 0.5 + 0.5) * 255;
			data[i + 1] = (ny * 0.5 + 0.5) * 255;
			data[i + 2] = (nz * 0.5 + 0.5) * 255;
			data[i + 3] = 255;
		}
	}
	return data;
}

function dataTexture(data, w, h, srgb = false) {
	const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat);
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.magFilter = THREE.LinearFilter;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.generateMipmaps = true;
	tex.anisotropy = 8;
	if (srgb) tex.colorSpace = THREE.SRGBColorSpace;
	tex.needsUpdate = true;
	return tex;
}

// Light oak planks with grain, pores and a satin finish.
export function makeWoodTextures(size = 1024) {
	const color = new Uint8ClampedArray(size * size * 4);
	const rough = new Uint8ClampedArray(size * size * 4);
	const height = new Float32Array(size * size);
	const planks = 4;
	for (let y = 0; y < size; y++) {
		for (let x = 0; x < size; x++) {
			const u = x / size, v = y / size;
			const plank = Math.floor(u * planks);
			const pu = u * planks - plank;
			const seed = plank * 31 + 7;
			// long grain running along v
			const warp = fbm(pu * 3, v * 2, 4, seed, 3 * 1024, 2) * 2.2;
			const rings = Math.sin((pu * 9 + warp * 2.5 + vnoise(v * 6, plank, seed) * 1.5) * Math.PI * 2);
			const fine = fbm(pu * 60, v * 3, 3, seed + 3, 60 * 1024, 3);
			const pores = vnoise(pu * 420, v * 18, seed + 5, 420 * 1024, 18) > 0.82 ? 1 : 0;
			const tone = 0.55 + 0.18 * rings + 0.25 * (fine - 0.5) - pores * 0.12 + (vnoise(plank, 3, 9) - 0.5) * 0.18;
			const gap = Math.min(pu, 1 - pu) < 0.004 ? 0.45 : 1;
			const r = (196 + 40 * tone) * gap, g = (150 + 34 * tone) * gap, b = (104 + 26 * tone) * gap;
			const i = (y * size + x) * 4;
			color[i] = r * 0.92; color[i + 1] = g * 0.86; color[i + 2] = b * 0.78; color[i + 3] = 255;
			const rr = 150 + 40 * (fine - 0.5) + pores * 50 + (gap < 1 ? 60 : 0);
			rough[i] = rough[i + 1] = rough[i + 2] = rr; rough[i + 3] = 255;
			height[y * size + x] = rings * 0.15 + fine * 0.5 - pores * 0.8 - (gap < 1 ? 3 : 0);
		}
	}
	return {
		map: dataTexture(color, size, size, true),
		roughnessMap: dataTexture(rough, size, size),
		normalMap: dataTexture(heightToNormal(height, size, size, 1.6), size, size),
	};
}

// Bright, airy cafe interior, painted already out of focus.
export function makeBackdropTexture(w = 1024, h = 1024) {
	const c = canvas(w, h);
	const g = c.getContext('2d');
	const grad = g.createLinearGradient(0, 0, w, 0);
	grad.addColorStop(0, '#f4ead9');
	grad.addColorStop(0.45, '#e8d9c2');
	grad.addColorStop(1, '#cdb79b');
	g.fillStyle = grad;
	g.fillRect(0, 0, w, h);
	g.filter = 'blur(28px)';
	// big window on the left with mullions
	g.fillStyle = '#fffaf0';
	g.fillRect(w * 0.02, h * 0.05, w * 0.42, h * 0.62);
	g.fillStyle = 'rgba(190,175,150,0.8)';
	g.fillRect(w * 0.22, h * 0.05, w * 0.025, h * 0.62);
	g.fillRect(w * 0.02, h * 0.34, w * 0.42, h * 0.02);
	// shelf and jars on the right
	g.fillStyle = 'rgba(120,86,56,0.85)';
	g.fillRect(w * 0.55, h * 0.30, w * 0.45, h * 0.025);
	const rnd = mulberry32(5);
	for (let i = 0; i < 7; i++) {
		g.fillStyle = `rgba(${150 + rnd() * 80},${110 + rnd() * 60},${70 + rnd() * 40},0.85)`;
		const x = w * (0.58 + i * 0.06), hh = h * (0.06 + rnd() * 0.08);
		g.fillRect(x, h * 0.30 - hh, w * 0.04, hh);
	}
	// plants
	for (let i = 0; i < 14; i++) {
		g.fillStyle = `rgba(${70 + rnd() * 40},${110 + rnd() * 50},${60 + rnd() * 30},0.75)`;
		g.beginPath();
		g.ellipse(w * (0.05 + rnd() * 0.3), h * (0.55 + rnd() * 0.18), w * (0.03 + rnd() * 0.05), h * (0.03 + rnd() * 0.06), rnd() * 3, 0, Math.PI * 2);
		g.fill();
	}
	for (let i = 0; i < 8; i++) {
		g.fillStyle = `rgba(${60 + rnd() * 40},${105 + rnd() * 40},${55 + rnd() * 30},0.8)`;
		g.beginPath();
		g.ellipse(w * (0.86 + rnd() * 0.12), h * (0.4 + rnd() * 0.2), w * 0.04, h * 0.05, rnd() * 3, 0, Math.PI * 2);
		g.fill();
	}
	// counter edge / lower wall
	g.fillStyle = 'rgba(160,128,96,0.9)';
	g.fillRect(0, h * 0.74, w, h * 0.26);
	g.filter = 'blur(10px)';
	// warm pendant lamps -> bokeh balls
	for (let i = 0; i < 5; i++) {
		const x = w * (0.55 + i * 0.1), y = h * (0.1 + (i % 2) * 0.05);
		const rg = g.createRadialGradient(x, y, 0, x, y, w * 0.05);
		rg.addColorStop(0, 'rgba(255,236,190,1)');
		rg.addColorStop(0.6, 'rgba(255,214,150,0.7)');
		rg.addColorStop(1, 'rgba(255,200,140,0)');
		g.fillStyle = rg;
		g.beginPath(); g.arc(x, y, w * 0.05, 0, Math.PI * 2); g.fill();
	}
	const tex = new THREE.CanvasTexture(c);
	tex.colorSpace = THREE.SRGBColorSpace;
	return tex;
}

// Condensation: dense micro-droplets, some medium beads and a few large drops.
export function makeCondensationNormal(w = 1024, h = 1024) {
	const height = new Float32Array(w * h);
	const rnd = mulberry32(42);
	const drop = (cx, cy, r) => {
		const r2 = r * r;
		for (let y = Math.floor(cy - r); y <= Math.ceil(cy + r); y++) {
			for (let x = Math.floor(cx - r); x <= Math.ceil(cx + r); x++) {
				const dx = x - cx, dy = (y - cy) * 0.92;
				const d2 = dx * dx + dy * dy;
				if (d2 > r2) continue;
				const hgt = Math.sqrt(r2 - d2);
				const xi = ((x % w) + w) % w, yi = ((y % h) + h) % h;
				const k = yi * w + xi;
				if (hgt > height[k]) height[k] = hgt;
			}
		}
	};
	for (let i = 0; i < 26000; i++) drop(rnd() * w, rnd() * h, 0.8 + rnd() * 1.8);
	for (let i = 0; i < 2600; i++) drop(rnd() * w, rnd() * h, 2.5 + rnd() * 3.5);
	for (let i = 0; i < 160; i++) drop(rnd() * w, rnd() * h, 6 + rnd() * 7);
	// a few drips that have run down, leaving clear trails
	for (let i = 0; i < 10; i++) {
		const x = rnd() * w, y0 = rnd() * h * 0.6, len = h * (0.15 + rnd() * 0.3);
		for (let y = y0; y < y0 + len; y += 2) drop(x + Math.sin(y * 0.03) * 2, y, 2.2);
		drop(x, y0 + len, 8 + rnd() * 3);
	}
	return dataTexture(heightToNormal(height, w, h, 0.55), w, h);
}

// Gentle undulations of a half-melted ice surface.
export function makeIceNormal(size = 256) {
	const height = new Float32Array(size * size);
	for (let y = 0; y < size; y++) {
		for (let x = 0; x < size; x++) {
			height[y * size + x] = fbm(x / size * 4, y / size * 4, 4, 11, 4, 4) * 6 + fbm(x / size * 16, y / size * 16, 2, 23, 16, 16) * 1.2;
		}
	}
	return dataTexture(heightToNormal(height, size, size, 1.2), size, size);
}

export function makeRadialTexture(stops, size = 256) {
	const c = canvas(size, size);
	const g = c.getContext('2d');
	const rg = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
	for (const [o, col] of stops) rg.addColorStop(o, col);
	g.fillStyle = rg;
	g.fillRect(0, 0, size, size);
	const tex = new THREE.CanvasTexture(c);
	tex.colorSpace = THREE.SRGBColorSpace;
	return tex;
}

// Caustic pattern a filled glass throws onto the table (bright ring + filaments).
export function makeCausticTexture(size = 512) {
	const c = canvas(size, size);
	const g = c.getContext('2d');
	const img = g.createImageData(size, size);
	for (let y = 0; y < size; y++) {
		for (let x = 0; x < size; x++) {
			const u = x / size - 0.5, v = y / size - 0.5;
			const r = Math.hypot(u * 1.0, v * 1.0);
			const a = Math.atan2(v, u);
			const fil = Math.pow(Math.abs(Math.sin(a * 7 + fbm(r * 8, a, 3, 4) * 6)), 8);
			const ring = Math.exp(-Math.pow((r - 0.30) / 0.05, 2));
			const core = Math.exp(-Math.pow(r / 0.22, 2)) * 0.6;
			const fall = Math.max(0, 1 - r / 0.5);
			const val = Math.min(1, (ring * 0.9 + core + fil * 0.35 * core) * fall);
			const i = (y * size + x) * 4;
			img.data[i] = img.data[i + 1] = img.data[i + 2] = val * 255;
			img.data[i + 3] = 255;
		}
	}
	g.putImageData(img, 0, 0);
	const tex = new THREE.CanvasTexture(c);
	tex.colorSpace = THREE.SRGBColorSpace;
	return tex;
}
