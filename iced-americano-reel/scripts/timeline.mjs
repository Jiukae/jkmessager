// Builds build/timeline.json from the narration durations.
// Every scene is long enough for both its voice line and its on-screen action;
// the 3D scene, the captions and the audio mix all read this one file.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const durations = JSON.parse(fs.readFileSync(path.join(ROOT, 'build/voice/durations.json'), 'utf8'));
const lines = JSON.parse(fs.readFileSync(path.join(ROOT, 'audio/narration.json'), 'utf8'));

// min: shortest the visual action can be; lead: silence before the voice starts; tail: after it ends
const SCENES = [
	{ id: 'hook', min: 3.8, lead: 0.30, tail: 0.45, step: 0, title: '홈카페 아이스 아메리카노' },
	{ id: 'ice', min: 5.4, lead: 0.35, tail: 0.50, step: 1, title: '컵에 얼음 가득', tag: '🧊 얼음 가득' },
	{ id: 'water', min: 5.6, lead: 0.35, tail: 0.50, step: 2, title: '찬물 붓기', tag: '💧 찬물 150ml' },
	{ id: 'espresso', min: 6.4, lead: 0.45, tail: 0.50, step: 3, title: '에스프레소 추출', tag: '☕ 2샷 · 약 40ml' },
	{ id: 'pour', min: 6.6, lead: 0.35, tail: 0.50, step: 4, title: '샷 천천히 붓기', tag: '얼음 위로 천천히' },
	{ id: 'stir', min: 5.6, lead: 0.35, tail: 0.50, step: 5, title: '저어주면 완성', tag: '✨ 완성!' },
	{ id: 'outro', min: 4.4, lead: 0.30, tail: 0.90, step: 0, title: '저장하고 따라 해보세요' },
];

let t = 0;
const scenes = SCENES.map((s) => {
	const voice = durations[s.id];
	const text = lines.find((l) => l.id === s.id).text;
	const dur = Math.max(s.min, s.lead + voice + s.tail);
	const scene = { ...s, start: +t.toFixed(3), end: +(t + dur).toFixed(3), voiceStart: +(t + s.lead).toFixed(3), voiceDur: voice, text };
	t += dur;
	return scene;
});

const timeline = { fps: 30, width: 1080, height: 1920, duration: +t.toFixed(3), scenes };
fs.writeFileSync(path.join(ROOT, 'build/timeline.json'), JSON.stringify(timeline, null, 2));
for (const s of scenes) console.log(`${s.id.padEnd(9)} ${s.start.toFixed(2)} → ${s.end.toFixed(2)}  voice@${s.voiceStart.toFixed(2)} (${s.voiceDur.toFixed(2)}s)`);
console.log(`total ${t.toFixed(2)}s`);
