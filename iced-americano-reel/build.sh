#!/usr/bin/env bash
# Builds output/iced_americano_reel.mp4.
#   ./build.sh               full 1080x1920 render (~65 min on 4 CPU cores, software WebGL)
#   PREVIEW=1 ./build.sh     540x960, every 3rd frame (~7 min)
#   AUDIO_ONLY=1 ./build.sh  new voice / sound mix on the already rendered frames (~2 min)
#
# Narration engine: Google Cloud TTS when GOOGLE_TTS_API_KEY is set (Korean male voice,
# TTS_VOICE to change it), otherwise the offline VITS voice. TTS_ENGINE=vits forces offline.
set -euo pipefail
cd "$(dirname "$0")"
WORKERS=${WORKERS:-3}
MODELS=build/models

encode() {
	ffmpeg -y -v error -framerate 30 -i build/frames/f_%05d.jpg -i build/audio/mix.wav \
		-af "loudnorm=I=-14:TP=-1.0:LRA=9" -ar 48000 \
		-c:v libx264 -preset slow -crf 21 -tune film -profile:v high -pix_fmt yuv420p -movflags +faststart \
		-c:a aac -b:a 192k -shortest output/iced_americano_reel.mp4
	echo "output/iced_americano_reel.mp4"
}

npm install --no-audit --no-fund
pip install --break-system-packages -q "sherpa-onnx==1.12.40" soundfile scipy numpy requests

# offline Korean voice (fallback narrator + café murmur) and Korean ASR used for voice QA
mkdir -p "$MODELS" output
REL=https://github.com/k2-fsa/sherpa-onnx/releases/download
[ -d "$MODELS/vits-mimic3-ko_KO-kss_low" ] || curl -sSL "$REL/tts-models/vits-mimic3-ko_KO-kss_low.tar.bz2" | tar xj -C "$MODELS"
[ -d "$MODELS/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" ] || curl -sSL "$REL/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2" | tar xj -C "$MODELS"

if [ "${AUDIO_ONLY:-0}" = "1" ]; then
	# reuse the published picture: the timeline/physics it was rendered with are kept in output/,
	# every voice line is stretched to its existing slot and only the audio track is replaced
	mkdir -p build
	cp output/timeline.json output/events.json build/
	TTS_FIT=1 python3 audio/tts.py
	python3 audio/mix.py
	ffmpeg -y -v error -i output/iced_americano_reel.mp4 -i build/audio/mix.wav -map 0:v:0 -map 1:a:0 -c:v copy \
		-af "loudnorm=I=-14:TP=-1.0:LRA=9" -ar 48000 -c:a aac -b:a 192k -shortest -movflags +faststart build/remux.mp4
	mv build/remux.mp4 output/iced_americano_reel.mp4
	echo "output/iced_americano_reel.mp4"
	exit 0
fi

python3 audio/tts.py            # build/voice/*.wav + durations.json
node scripts/timeline.mjs       # build/timeline.json (scene lengths follow the voice)
node scripts/render.mjs --events
python3 audio/mix.py            # build/audio/mix.wav

if [ "${PREVIEW:-0}" = "1" ]; then
	node scripts/render.mjs --scale 0.5 --stride 3 --workers "$WORKERS" --out build/preview
	ffmpeg -y -v error -framerate 10 -pattern_type glob -i "build/preview/f_*.jpg" -i build/audio/mix.wav \
		-c:v libx264 -pix_fmt yuv420p -crf 23 -c:a aac -b:a 160k -shortest output/preview.mp4
	echo "output/preview.mp4"
	exit 0
fi

node scripts/render.mjs --workers "$WORKERS"
encode
cp build/timeline.json build/events.json output/
