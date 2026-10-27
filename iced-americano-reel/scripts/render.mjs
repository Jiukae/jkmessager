// Renders the reel frame by frame in headless Chromium (software WebGL).
//   node scripts/render.mjs --stills 1.0,7.5      -> build/stills/*.png
//   node scripts/render.mjs --workers 3           -> build/frames/f_00000.jpg ...
//   node scripts/render.mjs --from 120 --to 240   -> a frame range only
// Also writes build/events.json (physics contacts + cue times) for the audio mix.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
	if (a.startsWith('--')) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true]);
	return acc;
}, []));

const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.woff2': 'font/woff2', '.woff': 'font/woff' };
const server = http.createServer((req, res) => {
	const p = path.join(ROOT, decodeURIComponent(new URL(req.url, 'http://x').pathname));
	if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); res.end(); return; }
	res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] ?? 'application/octet-stream' });
	fs.createReadStream(p).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const SCALE = parseFloat(args.scale ?? 1);
const url = `http://127.0.0.1:${server.address().port}/scene/index.html?scale=${SCALE}`;

const timeline = JSON.parse(fs.readFileSync(path.join(ROOT, 'build/timeline.json'), 'utf8'));
const total = Math.round(timeline.duration * timeline.fps);

async function openPage() {
	const browser = await chromium.launch({
		args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--disable-gpu-driver-bug-workarounds'],
	});
	const page = await browser.newPage({ viewport: { width: timeline.width, height: timeline.height }, deviceScaleFactor: SCALE });
	page.on('console', (m) => { if (m.text().startsWith('profile') || m.text().startsWith('physics')) console.log(m.text()); if (m.type() === 'error' || m.type() === 'warning' || args.verbose) console.log(`[page] ${m.text()}`); });
	page.on('pageerror', (e) => console.log(`[page error] ${e.message}`));
	if (args.profile) await page.addInitScript(() => { window.PROFILE = true; });
	await page.goto(url);
	await page.waitForFunction(() => window.REEL?.ready === true, null, { timeout: 600000 });
	return { browser, page };
}

async function shoot(page, t, file, type) {
	const a = Date.now();
	await page.evaluate((tt) => window.REEL.renderAt(tt), t);
	const b = Date.now();
	await page.screenshot({ path: file, type, ...(type === 'jpeg' ? { quality: 94 } : {}) });
	if (args.profile) console.log(`  evaluate ${b - a} ms, screenshot ${Date.now() - b} ms`);
}

const t0 = Date.now();
if (args.events) {
	const { browser, page } = await openPage();
	const data = await page.evaluate(() => ({ events: window.REEL.events, cues: window.REEL.cues }));
	fs.writeFileSync(path.join(ROOT, 'build/events.json'), JSON.stringify(data));
	console.log(`wrote build/events.json (${data.events.length} contacts)`);
	await browser.close();
} else if (args.stills) {
	const { browser, page } = await openPage();
	console.log(`page ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
	fs.mkdirSync(path.join(ROOT, 'build/stills'), { recursive: true });
	for (const s of String(args.stills).split(',')) {
		const t = parseFloat(s);
		const a = Date.now();
		await shoot(page, t, path.join(ROOT, `build/stills/t_${t.toFixed(2)}.png`), 'png');
		console.log(`still ${t.toFixed(2)}s in ${((Date.now() - a) / 1000).toFixed(2)}s`);
	}
	await browser.close();
} else {
	const out = path.join(ROOT, args.out ?? 'build/frames');
	const stride = parseInt(args.stride ?? 1, 10);
	fs.mkdirSync(out, { recursive: true });
	const from = parseInt(args.from ?? 0, 10), to = Math.min(parseInt(args.to ?? total, 10), total);
	const workers = parseInt(args.workers ?? 2, 10);
	let done = 0, next = from;
	const todo = () => {
		while (next < to && fs.existsSync(path.join(out, `f_${String(next).padStart(5, '0')}.jpg`)) && !args.force) next += stride;
		if (next >= to) return -1;
		const f = next;
		next += stride;
		return f;
	};
	await Promise.all(Array.from({ length: workers }, async (_, w) => {
		const { browser, page } = await openPage();
		if (w === 0) {
			const data = await page.evaluate(() => ({ events: window.REEL.events, cues: window.REEL.cues }));
			fs.writeFileSync(path.join(ROOT, 'build/events.json'), JSON.stringify(data));
		}
		for (let f = todo(); f >= 0; f = todo()) {
			await shoot(page, f / timeline.fps, path.join(out, `f_${String(f).padStart(5, '0')}.jpg`), 'jpeg');
			done++;
			if (done % 20 === 0) {
				const el = (Date.now() - t0) / 1000;
				console.log(`${done}/${to - from} frames, ${(el / done).toFixed(2)} s/frame, eta ${((to - from - done) * el / done / 60).toFixed(1)} min`);
			}
		}
		await browser.close();
	}));
	console.log(`rendered ${done} frames in ${((Date.now() - t0) / 60000).toFixed(1)} min`);
}
server.close();
