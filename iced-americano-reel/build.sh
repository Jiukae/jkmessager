#!/usr/bin/env bash
# Builds output/iced_americano_reel.mp4 from scratch.
#   ./build.sh            full 1080x1920 render (~45-90 min on 4 CPU cores, software WebGL)
#   PREVIEW=1 ./build.sh  540x960, every 3rd frame (~7 min)
set -euo pipefail
cd "$(dirname "$0")"
WORKERS=${WORKERS:-3}
MODELS=build/models

npm install --no-audit --no-fund
pip install --break-system-packages -q "sherpa-onnx==1.12.40" soundfile scipy numpy

# offline Korean TTS voice + Korean ASR used to pick the clearest take of each line
mkdir -p "$MODELS"
REL=https://github.com/k2-fsa/sherpa-onnx/releases/download
[ -d "$MODELS/vits-mimic3-ko_KO-kss_low" ] || curl -sSL "$REL/tts-models/vits-mimic3-ko_KO-kss_low.tar.bz2" | tar xj -C "$MODELS"
[ -d "$MODELS/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" ] || curl -sSL "$REL/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2" | tar xj -C "$MODELS"

python3 audio/tts.py            # build/voice/*.wav + durations.json
node scripts/timeline.mjs       # build/timeline.json (scene lengths follow the voice)
node scripts/render.mjs --events
python3 audio/mix.py            # build/audio/mix.wav

mkdir -p output
if [ "${PREVIEW:-0}" = "1" ]; then
	node scripts/render.mjs --scale 0.5 --stride 3 --workers "$WORKERS" --out build/preview
	ffmpeg -y -v error -framerate 10 -pattern_type glob -i "build/preview/f_*.jpg" -i build/audio/mix.wav \
		-c:v libx264 -pix_fmt yuv420p -crf 23 -c:a aac -b:a 160k -shortest output/preview.mp4
	echo "output/preview.mp4"
	exit 0
fi

node scripts/render.mjs --workers "$WORKERS"
ffmpeg -y -v error -framerate 30 -i build/frames/f_%05d.jpg -i build/audio/mix.wav \
	-af "loudnorm=I=-14:TP=-1.0:LRA=9" -ar 48000 \
	-c:v libx264 -preset slow -crf 18 -profile:v high -pix_fmt yuv420p -movflags +faststart \
	-c:a aac -b:a 192k -shortest output/iced_americano_reel.mp4
echo "output/iced_americano_reel.mp4"
