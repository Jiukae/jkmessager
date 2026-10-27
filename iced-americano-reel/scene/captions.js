// Reels-style overlay: step badge, progress bar, ingredient tag and karaoke
// subtitles. Purely a function of time so every frame is reproducible.
import { clamp01, easeOut, smooth } from './noise.js';

const $ = (sel) => document.querySelector(sel);

function pop(el, t, t0, t1, { dy = 40, scale = 0.86 } = {}) {
	const a = easeOut((t - t0) / 0.32);
	const out = t1 === undefined ? 1 : 1 - smooth(t1 - 0.22, t1, t);
	const k = Math.min(a, out);
	// a little overshoot on entry
	const s = scale + (1 - scale) * a + Math.sin(clamp01((t - t0) / 0.4) * Math.PI) * 0.04;
	el.style.opacity = k.toFixed(3);
	el.style.transform = `translateY(${((1 - a) * dy).toFixed(1)}px) scale(${s.toFixed(3)})`;
	el.style.visibility = k > 0.001 ? 'visible' : 'hidden';
}

export class Captions {
	constructor(timeline) {
		this.tl = timeline;
		this.el = { hook: $('#hook'), step: $('#step'), no: $('#step .no'), title: $('#step .title'), tag: $('#tag'), sub: $('#sub'), outro: $('#outro'), progress: $('#progress') };
		this.bars = [...document.querySelectorAll('#progress i b')];
		this.subScene = null;
		this.words = [];
	}

	setSubtitle(scene) {
		if (this.subScene === scene.id) return;
		this.subScene = scene.id;
		this.el.sub.innerHTML = '';
		const words = scene.text.split(' ');
		const weights = words.map((w) => w.replace(/[^가-힣a-zA-Z0-9]/g, '').length + 0.6);
		const total = weights.reduce((a, b) => a + b, 0);
		let acc = 0;
		this.words = words.map((w, i) => {
			const span = document.createElement('span');
			span.textContent = w + (i < words.length - 1 ? ' ' : '');
			this.el.sub.appendChild(span);
			const start = acc / total;
			acc += weights[i];
			return { span, at: start };
		});
	}

	update(t) {
		const scene = this.tl.scenes.find((s) => t < s.end) ?? this.tl.scenes[this.tl.scenes.length - 1];
		const { el } = this;

		// title cards
		if (scene.id === 'hook') pop(el.hook, t, scene.start + 0.05, scene.end, { dy: 60 });
		else el.hook.style.visibility = 'hidden';
		if (scene.id === 'outro') pop(el.outro, t, scene.start + 0.15, undefined, { dy: 60 });
		else el.outro.style.visibility = 'hidden';

		// step badge + tag
		if (scene.step > 0) {
			el.no.textContent = `STEP ${scene.step}`;
			el.title.textContent = scene.title;
			pop(el.step, t, scene.start + 0.08, scene.end + 0.2, { dy: 30 });
			el.tag.textContent = scene.tag ?? '';
			pop(el.tag, t, scene.start + 0.55, scene.end - 0.1, { dy: 24, scale: 0.7 });
		} else {
			el.step.style.visibility = 'hidden';
			el.tag.style.visibility = 'hidden';
		}

		// progress across the five steps
		const steps = this.tl.scenes.filter((s) => s.step > 0);
		const showBar = scene.step > 0;
		el.progress.style.opacity = showBar ? '1' : '0';
		steps.forEach((s, i) => {
			const f = clamp01((t - s.start) / (s.end - s.start));
			this.bars[i].style.width = `${(f * 100).toFixed(2)}%`;
		});

		// karaoke subtitle for the narration
		this.setSubtitle(scene);
		const v0 = scene.voiceStart, v1 = scene.voiceStart + scene.voiceDur;
		const visible = t >= v0 - 0.12 && t <= v1 + 0.45;
		pop(el.sub, t, v0 - 0.12, v1 + 0.45, { dy: 18, scale: 0.94 });
		if (!visible) el.sub.style.visibility = 'hidden';
		const prog = clamp01((t - v0) / Math.max(0.1, scene.voiceDur * 0.94));
		for (const w of this.words) w.span.classList.toggle('on', prog > w.at + 0.001);
	}
}
