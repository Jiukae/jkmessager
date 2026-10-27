// Opaque props: espresso machine group head, drip tray, straw, beans, table, backdrop.
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { beanGeometry } from './geometry.js';
import { mulberry32 } from './noise.js';

export const steel = (rough = 0.22) => new THREE.MeshPhysicalMaterial({ color: 0xd4d7da, metalness: 1, roughness: rough, envMapIntensity: 1.1 });

function brushedTexture() {
	const c = document.createElement('canvas');
	c.width = 512; c.height = 512;
	const g = c.getContext('2d');
	g.fillStyle = '#808080';
	g.fillRect(0, 0, 512, 512);
	const rnd = mulberry32(3);
	for (let i = 0; i < 2600; i++) {
		const v = 110 + rnd() * 50;
		g.strokeStyle = `rgba(${v},${v},${v},0.35)`;
		g.lineWidth = 0.5 + rnd();
		const y = rnd() * 512;
		g.beginPath(); g.moveTo(0, y); g.lineTo(512, y + (rnd() - 0.5) * 3); g.stroke();
	}
	const t = new THREE.CanvasTexture(c);
	t.wrapS = t.wrapT = THREE.RepeatWrapping;
	return t;
}

export function buildMachine() {
	const g = new THREE.Group();
	const brushed = brushedTexture();
	const body = new THREE.MeshPhysicalMaterial({ color: 0xcfd2d4, metalness: 1, roughness: 0.32, roughnessMap: brushed, envMapIntensity: 1.0 });
	const chrome = steel(0.07);
	const black = new THREE.MeshPhysicalMaterial({ color: 0x0b0b0c, roughness: 0.35, clearcoat: 0.8, clearcoatRoughness: 0.2 });
	const add = (geo, mat, x, y, z, rx = 0, ry = 0, rz = 0) => {
		const m = new THREE.Mesh(geo, mat);
		m.position.set(x, y, z);
		m.rotation.set(rx, ry, rz);
		m.castShadow = true; m.receiveShadow = true;
		g.add(m);
		return m;
	};
	// machine body and front panel above the group head
	add(new RoundedBoxGeometry(34, 24, 22, 4, 1.2), body, 0, 31, -7);
	add(new RoundedBoxGeometry(30, 1.2, 1.0, 2, 0.4), chrome, 0, 19.4, 4.2);
	// group head
	add(new THREE.CylinderGeometry(3.9, 4.1, 3.2, 64), chrome, 0, 17.6, 0);
	add(new THREE.CylinderGeometry(4.15, 4.15, 0.5, 64), chrome, 0, 16.0, 0);
	// portafilter basket, spout block and the two spouts
	add(new THREE.CylinderGeometry(3.7, 3.3, 2.8, 64), chrome, 0, 14.4, 0);
	add(new THREE.TorusGeometry(3.7, 0.18, 12, 64), chrome, 0, 15.75, 0, Math.PI / 2);
	add(new THREE.CylinderGeometry(1.35, 1.0, 1.0, 32), chrome, 0, 12.55, 0);
	for (const s of [-1, 1]) {
		add(new THREE.CylinderGeometry(0.32, 0.26, 1.3, 20), chrome, s * 0.85, 11.6, 0, 0, 0, s * 0.18);
	}
	// handle: chrome collar + black grip pointing to camera right
	const dir = new THREE.Vector3(0.72, -0.06, 0.69).normalize();
	const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
	const collar = add(new THREE.CylinderGeometry(0.9, 1.05, 2.4, 32), chrome, 0, 0, 0);
	collar.quaternion.copy(q);
	collar.position.copy(new THREE.Vector3(0, 14.4, 0).addScaledVector(dir, 4.6));
	const grip = add(new THREE.CylinderGeometry(0.85, 1.05, 11, 32), black, 0, 0, 0);
	grip.quaternion.copy(q);
	grip.position.copy(new THREE.Vector3(0, 14.4, 0).addScaledVector(dir, 11.2));
	const cap = add(new THREE.SphereGeometry(0.85, 24, 16), black, 0, 0, 0);
	cap.position.copy(new THREE.Vector3(0, 14.4, 0).addScaledVector(dir, 16.7));
	// drip tray with grate
	add(new RoundedBoxGeometry(32, 2.4, 20, 3, 0.5), body, 0, 1.2, -1);
	const slat = new THREE.BoxGeometry(0.42, 0.25, 18);
	for (let i = -16; i <= 16; i++) add(slat, chrome, i * 0.95, 2.52, -1);
	const tray = new THREE.Mesh(new THREE.PlaneGeometry(31, 18.5), new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.6 }));
	tray.rotation.x = -Math.PI / 2; tray.position.set(0, 2.38, -1);
	g.add(tray);
	return g;
}

export function buildStraw(length) {
	const g = new THREE.Group();
	const mat = new THREE.MeshPhysicalMaterial({ color: 0x0d0d0e, roughness: 0.22, clearcoat: 1, clearcoatRoughness: 0.08 });
	const inner = new THREE.MeshStandardMaterial({ color: 0x050505, roughness: 0.6, side: THREE.BackSide });
	const outer = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, length, 32, 1, true), mat);
	const innerM = new THREE.Mesh(new THREE.CylinderGeometry(0.27, 0.27, length, 24, 1, true), inner);
	const rim = new THREE.Mesh(new THREE.RingGeometry(0.27, 0.34, 32), mat);
	rim.rotation.x = -Math.PI / 2; rim.position.y = length / 2;
	// geometry is centred; shift so the group origin is the bottom end
	for (const m of [outer, innerM, rim]) { m.position.y += length / 2; m.castShadow = true; g.add(m); }
	return g;
}

export function buildBeans(count, avoid) {
	const geo = beanGeometry();
	const mat = new THREE.MeshPhysicalMaterial({ color: 0x3b1f10, roughness: 0.38, clearcoat: 0.6, clearcoatRoughness: 0.35, sheen: 0.3, sheenColor: new THREE.Color(0.4, 0.25, 0.15) });
	const mesh = new THREE.InstancedMesh(geo, mat, count);
	const rnd = mulberry32(9);
	const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), s = new THREE.Vector3(), p = new THREE.Vector3();
	let placed = 0, guard = 0;
	while (placed < count && guard++ < 5000) {
		const a = rnd() * Math.PI * 2, r = 6 + rnd() * 18;
		p.set(Math.cos(a) * r - 2, 0, Math.sin(a) * r - 4);
		if (avoid(p)) continue;
		const sc = 0.95 + rnd() * 0.2;
		const flip = rnd() < 0.55;
		p.y = 0.3 * sc;
		e.set(flip ? 0 : Math.PI, rnd() * Math.PI * 2, (rnd() - 0.5) * 0.3);
		q.setFromEuler(e);
		s.set(sc, sc, sc);
		mesh.setMatrixAt(placed++, m4.compose(p, q, s));
	}
	mesh.count = placed;
	mesh.castShadow = true;
	mesh.receiveShadow = true;
	return mesh;
}
