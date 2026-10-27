// Geometry builders. Units are centimetres; every container stands on y = 0.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { fbm, mulberry32 } from './noise.js';

const V = (x, y) => new THREE.Vector2(x, y);

function arc(points, cx, cy, r, a0, a1, n) {
	for (let i = 0; i <= n; i++) {
		const a = a0 + (a1 - a0) * (i / n);
		points.push(V(cx + r * Math.cos(a), cy + r * Math.sin(a)));
	}
}

// A tumbler-like container described by its outer radius at the foot/rim,
// height, wall thickness and solid base thickness.
export class Vessel {
	constructor({ height, rBottom, rTop, wall, base, corner = 0.3, segments = 96 }) {
		Object.assign(this, { height, rBottom, rTop, wall, base, corner, segments });
	}

	rOut(y) { return this.rBottom + (this.rTop - this.rBottom) * (y / this.height); }
	rIn(y) { return this.rOut(y) - this.wall; }

	// outer skin: foot, outer wall and the rounded rim crest
	outerGeometry() {
		const { height: H, wall: w, corner: c } = this;
		const pts = [V(0, 0)];
		arc(pts, this.rBottom - c, c, c, -Math.PI / 2, 0, 8);
		const n = 24;
		for (let i = 1; i <= n; i++) {
			const y = c + (H - c) * (i / n);
			pts.push(V(this.rOut(y), y));
		}
		arc(pts, this.rTop - w / 2, H, w / 2, 0, Math.PI, 10);
		return new THREE.LatheGeometry(pts, this.segments);
	}

	// inner skin: inside wall down to the top of the solid base
	innerGeometry() {
		const { height: H, base: b } = this;
		const pts = [];
		const n = 24;
		const ci = 0.45;
		for (let i = 0; i <= n; i++) {
			const y = H - (H - b - ci) * (i / n);
			pts.push(V(this.rIn(y), y));
		}
		arc(pts, this.rIn(b + ci) - ci, b + ci, ci, 0, -Math.PI / 2, 8);
		pts.push(V(0, b));
		return new THREE.LatheGeometry(pts, this.segments);
	}

	// closed volume of the cavity (slightly inset), clipped in the shader at the liquid level
	liquidGeometry(inset = 0.02, top = this.height - 0.05) {
		const { base: b } = this;
		const ci = 0.45;
		const pts = [V(0, b + inset)];
		arc(pts, this.rIn(b + ci) - ci, b + ci, ci - inset, -Math.PI / 2, 0, 8);
		const n = 24;
		for (let i = 1; i <= n; i++) {
			const y = b + ci + (top - b - ci) * (i / n);
			pts.push(V(this.rIn(y) - inset, y));
		}
		pts.push(V(0, top));
		return new THREE.LatheGeometry(pts, this.segments);
	}
}

// Water carafe: round body, shoulder, narrow neck with a lip.
export class Carafe extends Vessel {
	constructor() {
		super({ height: 22, rBottom: 4.4, rTop: 1.9, wall: 0.22, base: 0.6, corner: 0.6, segments: 72 });
	}

	rOut(y) {
		if (y < 11) return 4.4 + Math.sin((y / 11) * Math.PI * 0.5) * 0.35;
		if (y < 17.5) { const t = (y - 11) / 6.5; return 4.75 + (1.55 - 4.75) * (0.5 - 0.5 * Math.cos(t * Math.PI)); }
		if (y < 21.4) return 1.55;
		return 1.55 + (y - 21.4) * 0.55;
	}
}

// Slightly irregular, softly rounded ice cube.
export function iceGeometry(size, seed) {
	let g = new RoundedBoxGeometry(size, size * (0.92 + mulberry32(seed)() * 0.1), size, 5, size * 0.17);
	g.deleteAttribute('uv');
	g = mergeVertices(g, 1e-4);
	const pos = g.attributes.position;
	const v = new THREE.Vector3();
	for (let i = 0; i < pos.count; i++) {
		v.fromBufferAttribute(pos, i);
		const n = fbm(v.x * 0.6 + seed, v.y * 0.6 + v.z * 0.6, 3, seed) - 0.5;
		v.multiplyScalar(1 + n * 0.07);
		pos.setXYZ(i, v.x, v.y, v.z);
	}
	g.computeVertexNormals();
	// box-projected uvs for the bump map
	const uv = new Float32Array(pos.count * 2);
	const nrm = g.attributes.normal;
	for (let i = 0; i < pos.count; i++) {
		const ax = Math.abs(nrm.getX(i)), ay = Math.abs(nrm.getY(i)), az = Math.abs(nrm.getZ(i));
		let a, b;
		if (ax > ay && ax > az) { a = pos.getZ(i); b = pos.getY(i); }
		else if (ay > az) { a = pos.getX(i); b = pos.getZ(i); }
		else { a = pos.getX(i); b = pos.getY(i); }
		uv[i * 2] = a / size + 0.5 + seed * 0.13;
		uv[i * 2 + 1] = b / size + 0.5;
	}
	g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
	return g;
}

// Roasted coffee bean: squashed ellipsoid with the centre crease.
export function beanGeometry() {
	const g = new THREE.SphereGeometry(1, 28, 20);
	const p = g.attributes.position;
	for (let i = 0; i < p.count; i++) {
		let x = p.getX(i), y = p.getY(i), z = p.getZ(i);
		y *= 0.62;
		if (y > 0) {
			const crease = Math.exp(-(x * x) / 0.012) * 0.28 * Math.min(1, y * 3) * (1 - z * z * 0.6);
			y -= crease;
		}
		p.setXYZ(i, x * 0.48, y * 0.5, z * 0.68);
	}
	g.computeVertexNormals();
	return g;
}

// Liquid stream along a falling arc. Rebuilt every frame from a few control points.
export function streamGeometry(points, radiusAt, radialSegments = 12, tubularSegments = 48) {
	const curve = new THREE.CatmullRomCurve3(points);
	const frames = curve.computeFrenetFrames(tubularSegments, false);
	const pos = [], nor = [], uv = [], idx = [];
	const P = new THREE.Vector3(), N = new THREE.Vector3();
	for (let i = 0; i <= tubularSegments; i++) {
		const u = i / tubularSegments;
		curve.getPointAt(u, P);
		const r = radiusAt(u);
		const Nn = frames.normals[i], B = frames.binormals[i];
		for (let j = 0; j <= radialSegments; j++) {
			const v = (j / radialSegments) * Math.PI * 2;
			const s = Math.sin(v), c = -Math.cos(v);
			N.set(c * Nn.x + s * B.x, c * Nn.y + s * B.y, c * Nn.z + s * B.z).normalize();
			nor.push(N.x, N.y, N.z);
			pos.push(P.x + r * N.x, P.y + r * N.y, P.z + r * N.z);
			uv.push(u, j / radialSegments);
		}
	}
	for (let i = 1; i <= tubularSegments; i++) {
		for (let j = 1; j <= radialSegments; j++) {
			const a = (radialSegments + 1) * (i - 1) + (j - 1);
			const b = (radialSegments + 1) * i + (j - 1);
			const c = (radialSegments + 1) * i + j;
			const d = (radialSegments + 1) * (i - 1) + j;
			idx.push(a, b, d, b, c, d);
		}
	}
	const g = new THREE.BufferGeometry();
	g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
	g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
	g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
	g.setIndex(idx);
	return g;
}
