// Choreography: every animated quantity is a pure function of time so frames
// can be rendered in any order, by any number of workers.
import * as THREE from 'three';
import { clamp01, easeInOut, easeOut, lerp, mulberry32, noise1, smooth } from './noise.js';

// normalized integral of a trapezoid flow rate (ramps up, flows, ramps down)
function flowProgress(t, t0, t1, ramp = 0.3) {
	const total = t1 - t0 - ramp;
	if (t <= t0) return 0;
	if (t >= t1) return 1;
	let a;
	if (t < t0 + ramp) a = ((t - t0) ** 2) / (2 * ramp);
	else if (t <= t1 - ramp) a = ramp / 2 + (t - t0 - ramp);
	else a = total - ((t1 - t) ** 2) / (2 * ramp);
	return a / total;
}

function flowRate(t, t0, t1, ramp = 0.3) {
	return smooth(t0, t0 + ramp, t) * (1 - smooth(t1 - ramp, t1, t));
}

const v3 = (x, y, z) => new THREE.Vector3(x, y, z);

export class Director {
	constructor(timeline, glass) {
		this.tl = timeline;
		this.glass = glass;
		this.S = Object.fromEntries(timeline.scenes.map((s) => [s.id, s]));
		this.duration = timeline.duration;
		const S = this.S;
		this.simStart = S.ice.start;
		this.strawLength = 21;
		this.base = glass.base;
		this.LW = 10.3; // level after the water
		this.LF = 12.1; // level after the espresso

		const rnd = mulberry32(77);
		this.iceDrops = [];
		const n = 13;
		for (let i = 0; i < n; i++) {
			const a = rnd() * Math.PI * 2, r = 0.4 + rnd() * 1.1;
			this.iceDrops.push({ t: S.ice.start + 0.45 + i * 0.235, x: Math.cos(a) * r, y: 22.5 + rnd() * 2, z: Math.sin(a) * r, size: 2.45 + rnd() * 0.32 });
		}

		this.water = { in0: S.water.start, t0: S.water.start + 0.55, t1: S.water.end - 0.8, out1: S.water.end };
		const e0 = S.espresso.start;
		this.espresso = { pumpOn: e0 + 0.55, drip0: e0 + 1.05, flow0: e0 + 1.55, flow1: S.espresso.end - 1.0 };
		this.espresso.pumpOff = this.espresso.flow1 + 0.12;
		this.pour = { in0: S.pour.start, t0: S.pour.start + 0.75, t1: S.pour.end - 1.35, out1: S.pour.end };
		this.stir = { in0: S.stir.start + 0.1, in1: S.stir.start + 0.85, s1: S.stir.end - 1.15, rest1: S.stir.end - 0.4 };
		this.stir.s0 = this.stir.in1;
		this.stirOmega = Math.PI * 2 * 0.85; // rad/s while stirring
	}

	sceneAt(t) {
		const sc = this.tl.scenes;
		for (const s of sc) if (t < s.end) return s;
		return sc[sc.length - 1];
	}

	// the hook shows the finished drink, so it samples the end state
	stateTime(t) {
		return this.sceneAt(t).id === 'hook' ? this.duration - 0.02 : t;
	}

	// ------------------------------------------------------------ liquids

	waterProgress(t) { return flowProgress(t, this.water.t0, this.water.t1, 0.35); }
	pourProgress(t) { return flowProgress(t, this.pour.t0, this.pour.t1, 0.4); }

	mainLevel(t) {
		if (t < this.water.t0) return -100;
		const w = this.waterProgress(t);
		const p = this.pourProgress(t);
		return lerp(this.base + 0.02, this.LW, w) + (this.LF - this.LW) * p;
	}

	mixAmount(t) {
		const { s0, s1 } = this.stir;
		return Math.pow(easeInOut((t - s0) / (s1 - s0 + 0.6)), 0.85);
	}

	stirRate(t) {
		const { s0, s1 } = this.stir;
		return smooth(s0 - 0.05, s0 + 0.35, t) * (1 - smooth(s1 - 0.3, s1 + 0.1, t));
	}

	// liquid rotation angle (keeps turning a little after the straw stops)
	swirlAngle(t) {
		const { s0, s1 } = this.stir;
		if (t <= s0) return 0;
		const steps = 120;
		const end = Math.min(t, s1 + 0.1);
		let a = 0;
		const dt = (end - s0) / steps;
		for (let i = 0; i < steps; i++) a += this.stirRate(s0 + (i + 0.5) * dt) * this.stirOmega * 0.8 * dt;
		if (t > s1 + 0.1) a += this.stirOmega * 0.8 * 0.15 * 1.2 * (1 - Math.exp(-(t - s1 - 0.1) / 1.2));
		return a;
	}

	swirl(t) {
		const { s1 } = this.stir;
		const r = this.stirRate(t);
		const tail = t > s1 ? 0.25 * Math.exp(-(t - s1) / 1.0) : 0;
		return 110 * Math.max(r, tail);
	}

	agitation(t) {
		return 2600 * flowRate(t, this.water.t0, this.water.t1, 0.35) + 900 * flowRate(t, this.pour.t0, this.pour.t1, 0.4) + 300 * this.stirRate(t);
	}

	// concentration-field uniforms for the americano
	coffeeState(t) {
		const level = this.mainLevel(t);
		const p = this.pourProgress(t);
		const mix = this.mixAmount(t);
		const settle = clamp01((t - this.pour.t1) / 6);
		return {
			level,
			concTop: 2.7 * p,
			layer: level - (0.9 + 3.2 * p) - settle * 1.4,
			layerSoft: 0.9 + settle * 0.6,
			concBase: 1.0 * p,
			mix,
			swirl: this.swirlAngle(t),
			plume: 1.8 * flowRate(t, this.pour.t0, this.pour.t1 + 0.6, 0.4),
			foam: p * (0.75 - 0.35 * mix),
		};
	}

	// ------------------------------------------------------------ props

	strawPose(t, out) {
		const { in0, in1, s0, s1, rest1 } = this.stir;
		if (t < in0) return false;
		const L = this.strawLength;
		const b = this.base;
		const insertDir = v3(0.08, 1, 0.04).normalize();
		const bottomIn = v3(2.0, b + 0.45, 0.7);
		const restBottom = v3(-1.5, b + 0.38, -1.3);
		const restDir = v3(4.6, 12.8, 2.3).normalize();
		const th0 = Math.atan2(0.7, 2.0);
		let bottom, top;
		if (t < in1) {
			const k = easeOut((t - in0) / (in1 - in0));
			bottom = v3(lerp(2.3, bottomIn.x, k), lerp(25, bottomIn.y, k), lerp(0.9, bottomIn.z, k));
			top = bottom.clone().addScaledVector(insertDir, L);
		} else {
			const stirAt = (tt) => {
				let th = th0;
				const steps = 60, dt = (Math.min(tt, s1) - s0) / steps;
				for (let i = 0; i < steps; i++) th += this.stirRate(s0 + (i + 0.5) * dt) * this.stirOmega * dt;
				const rb = 1.6, rt = 0.7;
				const bt = v3(Math.cos(th) * rb, b + 0.6 + 0.35 * Math.sin(th * 2), Math.sin(th) * rb);
				const tp = v3(Math.cos(th) * rt, 0, Math.sin(th) * rt);
				const dxz = Math.hypot(tp.x - bt.x, tp.z - bt.z);
				tp.y = bt.y + Math.sqrt(L * L - dxz * dxz);
				const w = smooth(s0, s0 + 0.5, tt);
				const topIn = bottomIn.clone().addScaledVector(insertDir, L);
				return [bt.lerp(bottomIn, 1 - w), tp.lerp(topIn, 1 - w)];
			};
			[bottom, top] = stirAt(t);
			if (t > s1) {
				const k = easeInOut((t - s1) / (rest1 - s1));
				const restTop = restBottom.clone().addScaledVector(restDir, L);
				bottom.lerp(restBottom, k);
				top.lerp(restTop, k);
			}
			top = bottom.clone().add(top.clone().sub(bottom).setLength(L));
		}
		out.a.copy(bottom);
		out.b.copy(top);
		return true;
	}

	// carafe: pivot at the pouring lip, tilt about z
	carafeState(t) {
		const { in0, t0, t1, out1 } = this.water;
		if (t < in0 - 0.1 || t > out1 + 0.1) return null;
		const enter = easeOut((t - in0) / (t0 - in0 + 0.1));
		const leave = easeInOut((t - t1) / (out1 - t1));
		const tiltIn = lerp(1.15, 1.92, enter);
		const tilt = t < t1 ? tiltIn + 0.12 * smooth(t0, t1, t) : lerp(2.04, 1.1, leave);
		const lip = v3(lerp(9, 1.7, enter) + leave * 8, lerp(26, 18.6, enter) + leave * 6, 0.3);
		const rate = flowRate(t, t0, t1, 0.35);
		return { lip, tilt, rate, fill: 1 - this.waterProgress(t) * 0.55 };
	}

	shotPourState(t) {
		const { in0, t0, t1, out1 } = this.pour;
		if (t < in0 - 0.1 || t > out1 + 0.1) return null;
		const enter = easeOut((t - in0) / (t0 - in0 + 0.05));
		const leave = easeInOut((t - t1 - 0.15) / (out1 - t1 - 0.15));
		const p = this.pourProgress(t);
		const tilt = t < t1 ? lerp(0.5, 1.45, enter) + 0.75 * p : lerp(2.2, 0.9, leave);
		const lip = v3(lerp(12, 2.3, enter) + leave * 10, lerp(23, 17.2, enter) + leave * 5, 0.5);
		return { lip, tilt, rate: flowRate(t, t0, t1, 0.4), remaining: 1 - p };
	}

	espressoState(t) {
		const e = this.espresso;
		const p = flowProgress(t, e.flow0 - 0.2, e.flow1 + 0.2, 0.5);
		return {
			pump: t > e.pumpOn && t < e.pumpOff,
			flow: flowRate(t, e.flow0, e.flow1, 0.5),
			drips: (t > e.drip0 && t < e.flow0 + 0.2) || (t > e.flow1 - 0.1 && t < e.flow1 + 1.2),
			fill: p,
			blond: smooth(e.flow1 - 1.6, e.flow1, t),
		};
	}

	// ------------------------------------------------------------ camera

	camera(t) {
		const s = this.sceneAt(t);
		const p = clamp01((t - s.start) / (s.end - s.start));
		const q = easeInOut(p);
		let pos, target, fov = 34, focus, aperture = 1300;
		switch (s.id) {
			case 'hook': {
				const a = lerp(-0.62, 0.2, q), r = lerp(48, 41, q);
				pos = v3(Math.sin(a) * r, lerp(22, 15, q), Math.cos(a) * r);
				target = v3(0, lerp(14.5, 13.5, q), 0);
				focus = v3(0, 8, 2);
				break;
			}
			case 'ice':
				pos = v3(lerp(2, -1.5, q), lerp(34, 31, q), lerp(42, 38, q));
				target = v3(0, lerp(11.5, 10.5, q), 0);
				focus = v3(0, 12, 1);
				break;
			case 'water':
				pos = v3(lerp(13, 9, q), lerp(17, 17.5, q), lerp(43, 38, q));
				target = v3(0.8, lerp(12, 11.5, q), 0);
				focus = v3(0.5, 11, 1.5);
				break;
			case 'espresso':
				pos = v3(lerp(21, 15, q), lerp(13, 11, q), lerp(31, 27, q));
				target = v3(0, lerp(9, 8, q), 0);
				focus = v3(0, 6.5, 1.5);
				break;
			case 'pour':
				pos = v3(lerp(12, 7, q), lerp(28, 24, q), lerp(38, 34, q));
				target = v3(1.8, lerp(16, 14.5, q), 0);
				focus = v3(0.8, 13.5, 1.5);
				break;
			case 'stir':
				pos = v3(lerp(6, 1, q), lerp(44, 39, q), lerp(24, 30, q));
				target = v3(0, lerp(10, 9, q), 0);
				focus = v3(0, 13, 0.5);
				break;
			default: {
				const a = lerp(0.42, -0.28, q), r = lerp(44, 54, q);
				pos = v3(Math.sin(a) * r, lerp(15, 21, q), Math.cos(a) * r);
				target = v3(0, lerp(10.5, 11.5, q), 0);
				focus = v3(0, 8, 2);
			}
		}
		// handheld micro-motion
		pos.x += noise1(t * 0.7, 1) * 0.18; pos.y += noise1(t * 0.6, 2) * 0.15;
		target.x += noise1(t * 0.9, 3) * 0.08; target.y += noise1(t * 0.8, 4) * 0.08;
		// punch-in on every cut
		const since = t - s.start;
		fov *= 1 + 0.07 * (1 - easeOut(since / 0.35));
		const toEnd = s.end - t;
		let whip = 0;
		if (s.id !== 'hook' && since < 0.13) whip = 1 - since / 0.13;
		if (toEnd < 0.09 && s.id !== 'outro') whip = Math.max(whip, 1 - toEnd / 0.09);
		return { pos, target, fov, focusDist: pos.distanceTo(focus), aperture, whip, whipDir: s.step % 2 ? 1 : -1, scene: s };
	}
}
