#!/usr/bin/env bash
set -e

# 1. Ensure a URL was passed
if [ -z "$1" ]; then
  echo "❌ Error: Please provide a YouTube URL."
  echo "Usage: ./run.sh <YOUTUBE_URL> [INTERVAL]"
  exit 1
fi

URL="$1"
INTERVAL="${2:-60}"

# 2. Extract video ID (value after ?v= or &v=)
VIDEO_ID=$(echo "$URL" | grep -oP '(?<=[?&]v=)[^&]+')

if [ -z "$VIDEO_ID" ]; then
  echo "❌ Could not extract Video ID from URL: $URL"
  exit 1
fi


OUTPUT_FILE="stream_${VIDEO_ID}.mp4"

echo "📹 Target Video ID : $VIDEO_ID"
echo "📁 Target Output   : $OUTPUT_FILE"
echo "⏱️ Frame Interval  : $INTERVAL sec"

# 3. Check if file exists, download if missing
if [ -f "$OUTPUT_FILE" ]; then
  echo "✅ Found existing file '$OUTPUT_FILE'. Skipping download."
else
  echo "📥 '$OUTPUT_FILE' not found. Starting download with yt-dlp..."
  yt-dlp -f "bestvideo[vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
    --js-runtimes node \
    --extractor-args "youtube:player_client=ios,android_vr" \
    --merge-output-format mp4 \
    --concurrent-fragments 4 \
    --retries 20 \
    --fragment-retries 20 \
    -o "$OUTPUT_FILE" \
    "$URL"
fi

# 4. Run analysis script
echo "⚡ Starting BJJ Video Analysis on '$OUTPUT_FILE'..."
python3 bjj_video_analysis.py \
  --url "$OUTPUT_FILE" \
  --names "Bernice Lim" \
  --interval "$INTERVAL" \
  --crop
