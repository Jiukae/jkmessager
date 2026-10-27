// Scene assembly + deterministic frame API used by scripts/render.mjs:
//   await window.REEL.ready; window.REEL.renderAt(seconds)
import * as THREE from 'three';
import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';
import hdrUrl from '/node_modules/@pmndrs/assets/hdri/apartment.exr.js';
import { Vessel, Carafe, iceGeometry, streamGeometry } from './geometry.js';
import { glassOuterMaterial, liquidMaterials, iceMaterial } from './materials.js';
import { makeWoodTextures, makeBackdropTexture, makeCondensationNormal, makeIceNormal, makeRadialTexture, makeCausticTexture } from './textures.js';
import { buildMachine, buildStraw, buildBeans } from './props.js';
import { Director } from './director.js';
import { simulate } from './physics.js';
import { Pipeline } from './post.js';
import { Captions } from './captions.js';
import { clamp01, lerp, smooth } from './noise.js';

const timeline = await (await fetch('/build/timeline.json')).json();
const SCALE = parseFloat(new URLSearchParams(location.search).get('scale') ?? '1');
const W = Math.round(timeline.width * SCALE), H = Math.round(timeline.height * SCALE);

const canvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true, alpha: false, powerPreference: 'high-performance' });
renderer.setPixelRatio(1);
renderer.setSize(W, H, false);
renderer.autoClear = false;
renderer.toneMapping = THREE.NoToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.shadowMap.autoUpdate = false;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(34, W / H, 1.5, 700);
const pipeline = new Pipeline(renderer, W, H);
const sceneTexA = { value: pipeline.sceneA.texture };
const sceneTexB = { value: pipeline.sceneB.texture };
const groups = { A: [], B: [], C: [] };
const inPass = (k, ...objs) => { groups[k].push(...objs); return objs[0]; };

// ------------------------------------------------------------ lighting

{
	const exr = await new EXRLoader().loadAsync(hdrUrl);
	exr.mapping = THREE.EquirectangularReflectionMapping;
	const env = new THREE.Scene();
	env.background = exr;
	env.backgroundIntensity = 0.75;
	const box = (w, h, pos, intensity, color = [1, 0.97, 0.92]) => {
		const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(...color).multiplyScalar(intensity), side: THREE.DoubleSide }));
		m.position.set(...pos);
		m.lookAt(0, 10, 0);
		env.add(m);
	};
	box(70, 90, [-80, 35, -70], 5.5);         // big window behind-left
	box(16, 90, [75, 25, 15], 7, [1, 0.98, 0.95]);  // strip softbox right -> long highlight on the glass
	box(12, 70, [-70, 20, 45], 3.5);          // left kicker
	box(60, 60, [0, 95, 0], 1.6);             // ceiling
	box(80, 40, [0, 15, 110], 0.7, [1, 0.95, 0.88]); // soft fill from the camera side
	const pmrem = new THREE.PMREMGenerator(renderer);
	scene.environment = pmrem.fromScene(env, 0, 1, 1000).texture;
	pmrem.dispose();

	const sun = new THREE.DirectionalLight(new THREE.Color(1.0, 0.94, 0.86), 2.6);
	sun.position.set(-35, 50, -40);
	sun.target.position.set(0, 5, 0);
	sun.castShadow = true;
	sun.shadow.mapSize.set(2048, 2048);
	Object.assign(sun.shadow.camera, { left: -34, right: 34, top: 34, bottom: -34, near: 5, far: 200 });
	sun.shadow.bias = -0.0004;
	sun.shadow.normalBias = 0.04;
	scene.add(sun, sun.target);
}

// ------------------------------------------------------------ room

const wood = makeWoodTextures(1024);
for (const k of ['map', 'roughnessMap', 'normalMap']) wood[k].repeat.set(3.2, 2.2);
const table = new THREE.Mesh(new THREE.PlaneGeometry(240, 170), new THREE.MeshPhysicalMaterial({
	...wood, roughness: 1, normalScale: new THREE.Vector2(0.35, 0.35), clearcoat: 0.35, clearcoatRoughness: 0.32, envMapIntensity: 0.9,
}));
table.rotation.x = -Math.PI / 2;
table.position.set(0, 0, -30);
table.receiveShadow = true;
scene.add(inPass('A', table));

const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(460, 300), new THREE.MeshBasicMaterial({ map: makeBackdropTexture(), color: new THREE.Color(1.05, 1.0, 0.95) }));
backdrop.position.set(-20, 70, -115);
scene.add(inPass('A', backdrop));

// ------------------------------------------------------------ main set

const mainSet = new THREE.Group();
scene.add(mainSet);
const glass = new Vessel({ height: 14, rBottom: 3.45, rTop: 4.15, wall: 0.22, base: 1.1, corner: 0.35 });
const director = new Director(timeline, glass);

const condensation = makeCondensationNormal();
condensation.repeat.set(2, 1.3);
const mainOuterMat = glassOuterMaterial(sceneTexB, { wall: glass.wall, base: glass.base, normalMap: condensation });
const mainOuter = new THREE.Mesh(glass.outerGeometry(), mainOuterMat);
const mainInner = new THREE.Mesh(glass.innerGeometry(), glassOuterMaterial(sceneTexA, { wall: glass.wall, base: -1 }));
mainSet.add(inPass('C', mainOuter), inPass('B', mainInner));

function liquidMeshes(geo, liquid, parent) {
	const back = new THREE.Mesh(geo, liquid.back);
	const front = new THREE.Mesh(geo, liquid.front);
	parent.add(inPass('B', back), inPass('B', front));
	return [back, front];
}

const mainLiquid = liquidMaterials(sceneTexA);
mainLiquid.u.uRadius.value.set(glass.rBottom - glass.wall, (glass.rTop - glass.rBottom) / glass.height);
mainLiquid.u.uBottom.value = glass.base;
mainLiquid.u.uFoamColor.value.setRGB(0.5, 0.34, 0.2);
liquidMeshes(glass.liquidGeometry(), mainLiquid, mainSet);

// ice
const iceNormal = makeIceNormal();
const ice = director.iceDrops.map((d, i) => {
	const m = new THREE.Mesh(iceGeometry(d.size, i + 1), iceMaterial(iceNormal, i * 1.37));
	m.userData.size = d.size;
	mainSet.add(inPass('A', m));
	return m;
});

// straw
const straw = buildStraw(director.strawLength);
mainSet.add(inPass('A', straw));

// decals: contact occlusion, coloured glass shadow and caustic
const decal = (tex, w, h, opts) => {
	const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, ...opts }));
	m.rotation.x = -Math.PI / 2;
	m.renderOrder = 1;
	mainSet.add(inPass('A', m));
	return m;
};
const shadowDir = new THREE.Vector2(35, 40).normalize();
const contact = decal(makeRadialTexture([[0, 'rgba(0,0,0,0.85)'], [0.42, 'rgba(0,0,0,0.6)'], [0.55, 'rgba(0,0,0,0.12)'], [1, 'rgba(0,0,0,0)']]), 10.5, 10.5, { color: 0x000000, opacity: 0.55 });
contact.position.set(0, 0.02, 0);
const glassShadow = decal(makeRadialTexture([[0, 'rgba(255,255,255,1)'], [0.5, 'rgba(255,255,255,0.55)'], [1, 'rgba(255,255,255,0)']]), 8.5, 22, { color: new THREE.Color(0.1, 0.06, 0.035), opacity: 0.2 });
glassShadow.position.set(shadowDir.x * 8, 0.025, shadowDir.y * 8);
glassShadow.rotation.z = Math.atan2(shadowDir.x, shadowDir.y);
const caustic = decal(makeCausticTexture(), 7.5, 10, { blending: THREE.AdditiveBlending, color: new THREE.Color(0.3, 0.29, 0.27) });
caustic.position.set(shadowDir.x * 6.2, 0.03, shadowDir.y * 6.2);
caustic.rotation.z = glassShadow.rotation.z;

const beans = buildBeans(26, (p) => Math.hypot(p.x, p.z) < 7.5 || (p.z > 3 && Math.abs(p.x) < 9) || (p.z > 0 && p.x > 2 && p.x < 22 && p.z < 24));
mainSet.add(inPass('A', beans));

// water carafe (pivot at its pouring lip)
const carafe = new Carafe();
const carafePivot = new THREE.Group();
const carafeBody = new THREE.Group();
carafeBody.position.set(carafe.rTop - carafe.wall / 2, -carafe.height, 0);
carafePivot.add(carafeBody);
mainSet.add(carafePivot);
carafeBody.add(
	inPass('C', new THREE.Mesh(carafe.outerGeometry(), glassOuterMaterial(sceneTexB, { wall: carafe.wall, base: carafe.base }))),
	inPass('B', new THREE.Mesh(carafe.innerGeometry(), glassOuterMaterial(sceneTexA, { wall: carafe.wall, base: -1 }))),
);
const carafeLiquid = liquidMaterials(sceneTexA, { uniforms: { uSimple: { value: 1 }, uThick: { value: 4.5 } } });
liquidMeshes(carafe.liquidGeometry(), carafeLiquid, carafeBody);

const waterStreamMat = liquidMaterials(sceneTexA, { uniforms: { uSimple: { value: 1 }, uThick: { value: 0.5 }, uLevel: { value: 1e4 } } });
const waterStream = new THREE.Mesh(new THREE.BufferGeometry(), waterStreamMat.front);
mainSet.add(inPass('B', waterStream));

// shot glass used for pouring (pivot at the left-hand lip, tilts towards the glass on its left)
const shot = new Vessel({ height: 6.5, rBottom: 2.0, rTop: 2.45, wall: 0.18, base: 0.7, corner: 0.25, segments: 72 });
const espressoLinear = new THREE.Color(0.075, 0.03, 0.011);
function shotGlassGroup(parent, texA, texB, liquidOpts) {
	const g = new THREE.Group();
	g.add(
		inPass('C', new THREE.Mesh(shot.outerGeometry(), glassOuterMaterial(texB, { wall: shot.wall, base: shot.base }))),
		inPass('B', new THREE.Mesh(shot.innerGeometry(), glassOuterMaterial(texA, { wall: shot.wall, base: -1 }))),
	);
	const liquid = liquidMaterials(texA, { uniforms: liquidOpts });
	liquid.u.uConcBase.value = 6.5;
	liquid.u.uMix.value = 1;
	liquid.u.uFoamColor.value.setRGB(0.6, 0.36, 0.15);
	liquid.u.uRadius.value.set(shot.rBottom - shot.wall, (shot.rTop - shot.rBottom) / shot.height);
	liquid.u.uBottom.value = shot.base;
	liquidMeshes(shot.liquidGeometry(), liquid, g);
	parent.add(g);
	return { group: g, liquid };
}
const shotPivot = new THREE.Group();
mainSet.add(shotPivot);
const pourShot = shotGlassGroup(shotPivot, sceneTexA, sceneTexB, { uSimple: { value: 1 }, uThick: { value: 2.2 } });
pourShot.group.position.set(shot.rTop - shot.wall / 2, -shot.height, 0);

const streamMat = new THREE.MeshPhysicalMaterial({ color: espressoLinear, roughness: 0.1, clearcoat: 1, clearcoatRoughness: 0.05, emissive: new THREE.Color(0.02, 0.006, 0.001) });
const coffeeStream = new THREE.Mesh(new THREE.BufferGeometry(), streamMat);
mainSet.add(inPass('A', coffeeStream));

// ------------------------------------------------------------ espresso set

const machineSet = new THREE.Group();
scene.add(machineSet);
machineSet.add(inPass('A', buildMachine()));
const machineShot = shotGlassGroup(machineSet, sceneTexA, sceneTexB, {});
machineShot.group.position.set(0, 2.65, 0);
machineShot.liquid.u.uCenter.value.set(0, 2.65, 0);
const espressoStreams = [-1, 1].map(() => {
	const m = new THREE.Mesh(new THREE.BufferGeometry(), streamMat.clone());
	machineSet.add(inPass('A', m));
	return m;
});
const droplets = Array.from({ length: 6 }, () => {
	const m = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 12), streamMat);
	machineSet.add(inPass('A', m));
	return m;
});

// ------------------------------------------------------------ physics

const t0 = performance.now();
const sim = simulate(director, glass, director.iceDrops);
console.log(`physics: ${(performance.now() - t0).toFixed(0)} ms, ${sim.events.length} contacts`);

// ------------------------------------------------------------ per-frame state

const tmpP = new THREE.Vector3(), tmpQ = new THREE.Quaternion();
const up = new THREE.Vector3(0, 1, 0);
const strawPose = { a: new THREE.Vector3(), b: new THREE.Vector3() };

function setStream(mesh, pts, radiusAt) {
	mesh.geometry.dispose();
	mesh.geometry = streamGeometry(pts, radiusAt, 14, 56);
}

// ballistic stream from a lip; returns landing point or null
function updatePourStream(mesh, lip, mouthDir, speed, rate, tStart, tEnd, te, radius, landY) {
	const v0 = mouthDir.clone().multiplyScalar(speed);
	const g = 981;
	const sLand = (v0.y + Math.sqrt(v0.y * v0.y + 2 * g * Math.max(lip.y - landY, 0.01))) / g;
	const sHead = Math.min(te - tStart, sLand);
	const sTail = Math.max(0, te - tEnd);
	if (te < tStart || sHead <= sTail + 1e-3) { mesh.userData.active = false; return null; }
	const at = (s) => new THREE.Vector3(lip.x + v0.x * s, lip.y + v0.y * s - 0.5 * g * s * s, lip.z + v0.z * s);
	const pts = [];
	const n = 10;
	for (let i = 0; i <= n; i++) pts.push(at(sTail + (sHead - sTail) * (i / n)));
	const vLip = Math.max(speed, 12);
	setStream(mesh, pts, (u) => {
		const s = sTail + (sHead - sTail) * u;
		const v = Math.hypot(v0.x, v0.y - g * s, v0.z);
		let r = radius * Math.sqrt(vLip / v) * Math.sqrt(Math.max(rate, 0.08));
		r *= 1 + 0.1 * Math.sin(s * 120 - te * 40);
		if (sHead < sLand) r *= Math.min(1, (1 - u) / 0.06 + 0.35);
		if (sTail > 0) r *= Math.min(1, u / 0.06 + 0.35);
		return r;
	});
	mesh.userData.active = true;
	return sHead >= sLand - 1e-4 ? at(sLand) : null;
}

// top of the ice pile around (x, z), so a stream stops where it hits the ice
function iceTopNear(x, z, te) {
	let top = -100;
	for (let i = 0; i < ice.length; i++) {
		if (!sim.pose(i, te, tmpP, tmpQ)) continue;
		if (Math.hypot(tmpP.x - x, tmpP.z - z) < 1.9) top = Math.max(top, tmpP.y + ice[i].userData.size * 0.5);
	}
	return top;
}

function updateWorld(t) {
	const te = director.stateTime(t);
	const sc = director.sceneAt(t);
	mainSet.visible = sc.id !== 'espresso';
	machineSet.visible = sc.id === 'espresso';

	// americano
	const cs = director.coffeeState(te);
	const u = mainLiquid.u;
	u.uTime.value = t;
	u.uLevel.value = cs.level;
	u.uConcTop.value = cs.concTop;
	u.uLayer.value = cs.layer;
	u.uLayerSoft.value = cs.layerSoft;
	u.uConcBase.value = cs.concBase;
	u.uMix.value = cs.mix;
	u.uSwirl.value = cs.swirl;
	u.uFoam.value = cs.foam;
	u.uRipple.value.z = 0;
	u.uPlume.value.z = 0;
	u.uTurb.value = 0.02 + 0.12 * director.stirRate(te) + 0.05 * Math.max(0, 1 - Math.abs(te - director.pour.t1) / 2) * cs.concTop;

	// ice
	ice.forEach((m, i) => {
		m.userData.active = sim.pose(i, te, m.position, m.quaternion);
	});

	// condensation builds up once the glass is cold
	const cond = smooth(director.S.ice.start + 1.2, director.S.water.end + 2, te);
	mainOuterMat.normalScale.set(0.42 * cond, 0.42 * cond);
	mainOuterMat.userData.u.uFog.value = 0.35 * cond;
	mainOuterMat.roughness = 0.015 + 0.035 * cond;

	// straw
	straw.userData.active = director.strawPose(te, strawPose);
	if (straw.userData.active) {
		straw.position.copy(strawPose.a);
		straw.quaternion.setFromUnitVectors(up, strawPose.b.clone().sub(strawPose.a).normalize());
	}

	// shadow tint & caustic follow what is in the glass
	const coffee = Math.max(cs.concTop / 2.7, cs.concBase);
	const hasWater = cs.level > glass.base + 0.3 ? 1 : 0;
	glassShadow.material.opacity = 0.16 + 0.36 * coffee;
	caustic.material.color.setRGB(lerp(0.32, 0.5, coffee), lerp(0.31, 0.24, coffee), lerp(0.29, 0.07, coffee)).multiplyScalar(0.55 + 0.6 * hasWater - 0.25 * coffee);

	// carafe + water stream
	const cst = director.carafeState(te);
	carafePivot.visible = !!cst;
	waterStream.userData.active = false;
	if (cst) {
		carafePivot.position.copy(cst.lip);
		carafePivot.rotation.set(0, 0, cst.tilt);
		carafeLiquid.u.uLevel.value = cst.lip.y + 1.5 + 6 * cst.fill;
		carafeLiquid.u.uTime.value = t;
		const mouth = new THREE.Vector3(-Math.sin(cst.tilt), Math.cos(cst.tilt), 0);
		const land = Math.max(cs.level, iceTopNear(cst.lip.x - 1.5, cst.lip.z, te));
		const hit = updatePourStream(waterStream, cst.lip, mouth, 14, cst.rate + 0.15, director.water.t0, director.water.t1, te, 0.32, land);
		if (hit && hit.y <= cs.level + 0.2) { u.uRipple.value.set(hit.x, hit.z, 0.35 * cst.rate, 0); }
	}

	// shot glass pouring espresso
	const sst = director.shotPourState(te);
	shotPivot.visible = !!sst;
	coffeeStream.userData.active = false;
	if (sst) {
		shotPivot.position.copy(sst.lip);
		shotPivot.rotation.set(0, 0, sst.tilt);
		const pl = pourShot.liquid.u;
		pl.uTime.value = t;
		const bottomW = new THREE.Vector3(0, shot.base, 0);
		pourShot.group.localToWorld(bottomW);
		pl.uLevel.value = sst.remaining < 0.01 ? -100 : Math.min(sst.lip.y + 0.15 + 2.0 * sst.remaining, bottomW.y + (2.6 * sst.remaining) / Math.max(Math.cos(sst.tilt), 0.3));
		pl.uCrema.value = 0.08 + 0.2 * sst.remaining;
		const mouth = new THREE.Vector3(-Math.sin(sst.tilt), Math.cos(sst.tilt), 0);
		const land = Math.max(cs.level, iceTopNear(sst.lip.x - 1.2, sst.lip.z, te));
		const hit = updatePourStream(coffeeStream, sst.lip, mouth, 10, sst.rate + 0.1, director.pour.t0, director.pour.t1, te, 0.24, land);
		if (hit) {
			u.uPlume.value.set(hit.x, hit.z, cs.concTop > 0 ? 1.6 * sst.rate : 0, 0);
			if (hit.y <= cs.level + 0.2) u.uRipple.value.set(hit.x, hit.z, 0.25 * sst.rate, 0);
		}
	}

	// espresso extraction
	if (machineSet.visible) {
		const es = director.espressoState(te);
		const ml = machineShot.liquid.u;
		ml.uTime.value = t;
		const surface = 2.65 + shot.base + 0.05 + 2.35 * es.fill;
		ml.uLevel.value = es.fill > 0.004 ? surface : -100;
		ml.uCrema.value = 0.05 + 0.45 * es.fill;
		ml.uFoam.value = 1;
		ml.uRipple.value.set(0, 0, 0.15 * es.flow, 0);
		const col = new THREE.Color().lerpColors(espressoLinear, new THREE.Color(0.32, 0.15, 0.055), es.blond);
		espressoStreams.forEach((m, k) => {
			const s = k ? 1 : -1;
			m.material.color.copy(col);
			m.userData.active = es.flow > 0.02;
			if (!m.userData.active) return;
			const top = new THREE.Vector3(s * 0.97, 10.95, 0);
			const bot = new THREE.Vector3(s * 0.75, surface, 0.05 * s);
			const pts = [];
			for (let i = 0; i <= 8; i++) {
				const f = i / 8;
				const p = top.clone().lerp(bot, f);
				p.x += Math.sin(f * 5 + t * 6 + k) * 0.025 * f;
				p.z += Math.cos(f * 4 + t * 4 + k) * 0.025 * f;
				pts.push(p);
			}
			setStream(m, pts, (f) => (0.05 + 0.05 * es.flow) * (1.2 - 0.4 * f) * (1 + 0.05 * Math.sin(f * 30 - t * 24 + k)));
		});
		droplets.forEach((d, k) => {
			const s = k % 2 ? 1 : -1;
			const period = 0.42;
			const phase = (te + (k >> 1) * 0.14) / period;
			const born = Math.floor(phase) * period - (k >> 1) * 0.14;
			const age = te - born;
			const y = 10.85 - 0.5 * 981 * age * age;
			d.userData.active = es.drips && y > surface && es.flow < 0.6 && born > director.espresso.drip0;
			d.position.set(s * 0.95, y, 0);
			d.scale.set(1, 1 + age * 2.5, 1);
		});
	}
	return sc;
}

// ------------------------------------------------------------ frame API

const captions = new Captions(timeline);
await document.fonts.load('900 60px "Noto Sans KR"', '아이스 아메리카노 완성 저장');
await document.fonts.load('700 40px "Noto Sans KR"', '홈카페 레시피 단계');
await document.fonts.ready;

function renderAt(t) {
	const sc = updateWorld(t);
	const cam = director.camera(t);
	camera.position.copy(cam.pos);
	camera.fov = cam.fov;
	camera.lookAt(cam.target);
	camera.updateProjectionMatrix();
	camera.updateMatrixWorld();
	const fadeIn = smooth(0, 0.18, t), fadeOut = 1 - smooth(timeline.duration - 0.5, timeline.duration, t);
	pipeline.render(scene, camera, groups, {
		aperture: cam.aperture * SCALE, focusDist: cam.focusDist, whip: cam.whip, whipDir: cam.whipDir, time: t,
		exposure: 1.0, fade: Math.min(fadeIn, fadeOut) * 0.85 + 0.15 * clamp01(t * 10),
	});
	captions.update(t);
	return { scene: sc.id };
}

window.REEL = {
	timeline,
	events: sim.events,
	cues: {
		iceDrops: director.iceDrops.map((d) => d.t),
		water: director.water, espresso: director.espresso, pour: director.pour, stir: director.stir,
	},
	renderAt,
	ready: true,
};
document.title = 'ready';
