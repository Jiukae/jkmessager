// Rigid-body simulation of the ice (cannon-es), run once up front at a fixed
// 240 Hz step so every render worker gets identical motion. Contact events are
// exported so the audio mix can place a clink on every real collision.
import * as CANNON from 'cannon-es';
import * as THREE from 'three';
import { mulberry32 } from './noise.js';

export const PHYS_HZ = 240;

export function simulate(director, glass, ice) {
	const world = new CANNON.World({ gravity: new CANNON.Vec3(0, -981, 0) });
	world.broadphase = new CANNON.NaiveBroadphase();
	world.solver.iterations = 30;
	world.solver.tolerance = 1e-4;
	world.allowSleep = false;
	world.defaultContactMaterial.contactEquationStiffness = 1e7;
	world.defaultContactMaterial.contactEquationRelaxation = 4;

	const matIce = new CANNON.Material('ice');
	const matGlass = new CANNON.Material('glass');
	const matStraw = new CANNON.Material('straw');
	world.addContactMaterial(new CANNON.ContactMaterial(matIce, matIce, { friction: 0.05, restitution: 0.12 }));
	world.addContactMaterial(new CANNON.ContactMaterial(matIce, matGlass, { friction: 0.04, restitution: 0.22 }));
	world.addContactMaterial(new CANNON.ContactMaterial(matIce, matStraw, { friction: 0.1, restitution: 0.05 }));

	// glass: solid floor + rings of wall slabs (extra invisible height keeps ice from escaping)
	const glassBody = new CANNON.Body({ mass: 0, material: matGlass });
	glassBody.userData = { kind: 'glass' };
	glassBody.addShape(new CANNON.Box(new CANNON.Vec3(5, 0.5, 5)), new CANNON.Vec3(0, glass.base - 0.5, 0));
	const N = 28;
	const bands = [[glass.base, 4.5], [4.5, 8], [8, 11.5], [11.5, 14], [14, 24]];
	for (const [y0, y1] of bands) {
		const ym = (y0 + y1) / 2;
		const r = glass.rIn(Math.min(ym, glass.height)) - 0.04;
		const w = ((2 * Math.PI * r) / N) * 1.2;
		for (let i = 0; i < N; i++) {
			const th = (i / N) * Math.PI * 2;
			const q = new CANNON.Quaternion();
			q.setFromAxisAngle(new CANNON.Vec3(0, 1, 0), Math.PI / 2 - th);
			glassBody.addShape(new CANNON.Box(new CANNON.Vec3(w / 2, (y1 - y0) / 2, 0.5)),
				new CANNON.Vec3(Math.cos(th) * (r + 0.5), ym, Math.sin(th) * (r + 0.5)), q);
		}
	}
	world.addBody(glassBody);

	const table = new CANNON.Body({ mass: 0, material: matGlass, shape: new CANNON.Plane() });
	table.quaternion.setFromAxisAngle(new CANNON.Vec3(1, 0, 0), -Math.PI / 2);
	table.userData = { kind: 'table' };
	world.addBody(table);

	const straw = new CANNON.Body({ mass: 0, type: CANNON.Body.KINEMATIC, material: matStraw });
	straw.addShape(new CANNON.Cylinder(0.42, 0.42, director.strawLength, 12));
	straw.userData = { kind: 'straw' };
	straw.position.set(0, -200, 0);
	world.addBody(straw);

	const rnd = mulberry32(2024);
	const bodies = ice.map((cube, i) => {
		const s = cube.size;
		const b = new CANNON.Body({ mass: 0.917 * s * s * s, material: matIce, linearDamping: 0.02, angularDamping: 0.05 });
		b.addShape(new CANNON.Box(new CANNON.Vec3(s * 0.47, s * 0.45, s * 0.47)));
		b.userData = { kind: 'ice', index: i, size: s };
		const drop = director.iceDrops[i];
		b.position.set(drop.x, drop.y, drop.z);
		b.quaternion.setFromEuler(rnd() * 6.28, rnd() * 6.28, rnd() * 6.28);
		b.velocity.set(0, -40, 0);
		b.angularVelocity.set((rnd() - 0.5) * 8, (rnd() - 0.5) * 8, (rnd() - 0.5) * 8);
		b.spawnTime = drop.t;
		b.added = false;
		return b;
	});

	const events = [];
	const lastPair = new Map();
	let simTime = director.simStart;
	const onCollide = (e) => {
		const a = e.target, b = e.body;
		const v = Math.abs(e.contact.getImpactVelocityAlongNormal());
		if (v < 12) return;
		const key = a.id < b.id ? `${a.id}:${b.id}` : `${b.id}:${a.id}`;
		const last = lastPair.get(key);
		if (last !== undefined && simTime - last < 0.05) return;
		lastPair.set(key, simTime);
		const level = director.mainLevel(simTime);
		events.push({
			t: +simTime.toFixed(4),
			v: +v.toFixed(1),
			kind: b.userData?.kind === 'ice' ? 'ice' : b.userData?.kind ?? 'glass',
			y: +a.position.y.toFixed(2),
			under: a.position.y < level - 0.5,
			level: +level.toFixed(2),
		});
	};
	bodies.forEach((b) => b.addEventListener('collide', onCollide));

	const dt = 1 / PHYS_HZ;
	const steps = Math.ceil((director.duration - director.simStart) * PHYS_HZ) + 2;
	const data = new Float32Array(steps * bodies.length * 7);
	const strawPose = { a: new THREE.Vector3(), b: new THREE.Vector3() };
	const nextPose = { a: new THREE.Vector3(), b: new THREE.Vector3() };
	const up = new THREE.Vector3(0, 1, 0), dir = new THREE.Vector3(), q3 = new THREE.Quaternion();

	for (let step = 0; step < steps; step++) {
		simTime = director.simStart + step * dt;
		const level = director.mainLevel(simTime);

		for (const b of bodies) {
			if (!b.added && simTime >= b.spawnTime) { world.addBody(b); b.added = true; }
		}

		// kinematic straw follows the director's choreography
		const active = director.strawPose(simTime, strawPose);
		if (active) {
			director.strawPose(simTime + dt, nextPose);
			const mid = strawPose.a.clone().add(strawPose.b).multiplyScalar(0.5);
			const midN = nextPose.a.clone().add(nextPose.b).multiplyScalar(0.5);
			dir.copy(strawPose.b).sub(strawPose.a).normalize();
			q3.setFromUnitVectors(up, dir);
			straw.position.set(mid.x, mid.y, mid.z);
			straw.quaternion.set(q3.x, q3.y, q3.z, q3.w);
			straw.velocity.set((midN.x - mid.x) / dt, (midN.y - mid.y) / dt, (midN.z - mid.z) / dt);
		} else {
			straw.position.set(0, -200, 0);
			straw.velocity.set(0, 0, 0);
		}

		const swirl = director.swirl(simTime);
		const agitation = director.agitation(simTime);
		for (const b of bodies) {
			if (!b.added) continue;
			const s = b.userData.size;
			const bottom = b.position.y - s * 0.55;
			const f = Math.min(1, Math.max(0, (level - bottom) / (s * 1.1)));
			if (f > 0) {
				const m = b.mass;
				// buoyancy (water / ice density ratio) and viscous drag
				b.force.y += (m * 981 * f) / 0.917;
				b.force.x -= m * 5 * f * b.velocity.x;
				b.force.y -= m * 5 * f * b.velocity.y;
				b.force.z -= m * 5 * f * b.velocity.z;
				b.torque.x -= m * 8 * f * b.angularVelocity.x;
				b.torque.y -= m * 8 * f * b.angularVelocity.y;
				b.torque.z -= m * 8 * f * b.angularVelocity.z;
				// swirling liquid drags the cubes around the axis
				if (swirl > 0) {
					const x = b.position.x, z = b.position.z;
					const r = Math.hypot(x, z) + 0.3;
					b.force.x += (-z / r) * m * swirl * f;
					b.force.z += (x / r) * m * swirl * f;
				}
				// pouring turbulence near the surface
				if (agitation > 0) {
					const near = Math.max(0, 1 - Math.abs(level - b.position.y) / 4);
					b.force.x += (rnd() - 0.5) * m * agitation * near;
					b.force.z += (rnd() - 0.5) * m * agitation * near;
					b.force.y += (rnd() - 0.4) * m * agitation * near;
				}
			}
		}

		world.step(dt);

		// hard containment: a cube squeezed between the straw and the wall must not tunnel out,
		// and nothing gets launched far above the liquid
		for (const b of bodies) {
			if (!b.added) continue;
			const s = b.userData.size;
			const y = Math.min(Math.max(b.position.y, glass.base), glass.height);
			const rMax = glass.rIn(y) - s * 0.42;
			const r = Math.hypot(b.position.x, b.position.z);
			if (r > rMax && b.position.y < glass.height + 6) {
				const nx = b.position.x / r, nz = b.position.z / r;
				b.position.x = nx * rMax;
				b.position.z = nz * rMax;
				const vr = b.velocity.x * nx + b.velocity.z * nz;
				if (vr > 0) { b.velocity.x -= vr * nx * 1.2; b.velocity.z -= vr * nz * 1.2; }
			}
			if (level > glass.base && b.position.y - level > s * 0.9 && b.velocity.y > 0) b.velocity.y *= 0.6;
		}

		const base = step * bodies.length * 7;
		bodies.forEach((b, i) => {
			const o = base + i * 7;
			const p = b.added ? b.position : { x: 0, y: -500, z: 0 };
			data[o] = p.x; data[o + 1] = p.y; data[o + 2] = p.z;
			data[o + 3] = b.quaternion.x; data[o + 4] = b.quaternion.y; data[o + 5] = b.quaternion.z; data[o + 6] = b.quaternion.w;
		});
	}

	return {
		events,
		count: bodies.length,
		// interpolated pose of cube i at time t
		pose(i, t, pos, quat) {
			const x = Math.min(Math.max((t - director.simStart) * PHYS_HZ, 0), steps - 1.001);
			const k = Math.floor(x), f = x - k;
			const o0 = (k * bodies.length + i) * 7, o1 = ((k + 1) * bodies.length + i) * 7;
			pos.set(data[o0] + (data[o1] - data[o0]) * f, data[o0 + 1] + (data[o1 + 1] - data[o0 + 1]) * f, data[o0 + 2] + (data[o1 + 2] - data[o0 + 2]) * f);
			const qa = new THREE.Quaternion(data[o0 + 3], data[o0 + 4], data[o0 + 5], data[o0 + 6]);
			const qb = new THREE.Quaternion(data[o1 + 3], data[o1 + 4], data[o1 + 5], data[o1 + 6]);
			quat.slerpQuaternions(qa, qb, f);
			return data[o0 + 1] > -400;
		},
	};
}
