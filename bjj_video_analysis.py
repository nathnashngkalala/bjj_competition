import argparse
import time
import os
import cv2
import easyocr
import subprocess
import json
import re
from tqdm import tqdm
from google import genai

# Auto-detect CUDA GPU availability if PyTorch is present
try:
    import torch
    use_gpu = torch.cuda.is_available()
except ImportError:
    use_gpu = False

print(f"⚡ Initializing EasyOCR (GPU: {use_gpu})...")
reader = easyocr.Reader(['en'], gpu=use_gpu)
CHECKPOINT_FILE = "scan_checkpoint.json"


def extract_youtube_id(url):
    """Extracts the 11-character YouTube video ID from standard or short URLs."""
    pattern = r"(?:v=|\/embed\/|\/v\/|youtu\.be\/|\/shorts\/|\/live\/|^)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None


def load_checkpoint(video_path, target_names):
    """Loads scan progress if a valid checkpoint exists for the current video."""
    default_data = {
        "video_path": video_path,
        "last_frame_idx": 0,
        "matched_timestamps": {name: [] for name in target_names},
        "frames_checked": 0
    }
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
            if data.get("video_path") == video_path:
                print(f"🔄 Checkpoint found! Resuming scan from frame {data.get('last_frame_idx', 0)}...")
                return data
        except Exception:
            pass
    return default_data


def save_checkpoint(video_path, last_frame_idx, matched_timestamps, frames_checked):
    """Saves current scan state to disk."""
    data = {
        "video_path": video_path,
        "last_frame_idx": last_frame_idx,
        "matched_timestamps": matched_timestamps,
        "frames_checked": frames_checked
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


def clear_checkpoint():
    """Removes checkpoint file upon successful completion."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


def cluster_matches(timestamps, max_gap_sec=120, buffer_sec=20, min_hits=2):
    """
    Groups individual detections into distinct matches per athlete.
    - `max_gap_sec`: Max gap between score detections before starting a new match (120s).
    - `buffer_sec`: Padding added before/after match start/end (20s).
    - `min_hits`: Minimum OCR detections required to filter out stray bracket graphics.
    """
    if not timestamps:
        return []

    sorted_ts = sorted(list(set(timestamps)))
    matches = []
    current_match = [sorted_ts[0]]

    for ts in sorted_ts[1:]:
        if ts - current_match[-1] > max_gap_sec:
            if len(current_match) >= min_hits:
                start = max(0, current_match[0] - buffer_sec)
                end = current_match[-1] + buffer_sec
                matches.append((start, end))
            current_match = [ts]
        else:
            current_match.append(ts)

    if len(current_match) >= min_hits:
        start = max(0, current_match[0] - buffer_sec)
        end = current_match[-1] + buffer_sec
        matches.append((start, end))

    return matches


def download_stream(youtube_url, output_path):
    """Downloads YouTube stream using yt-dlp to a specific filename."""
    print(f"\n[1/4] 📥 Downloading YouTube Stream to '{output_path}'...")
    start_time = time.perf_counter()

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "-o", output_path,
        youtube_url
    ]
    subprocess.run(cmd, check=True)

    elapsed = time.perf_counter() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Download Complete: {file_size_mb:.2f} MB in {elapsed:.2f}s ({file_size_mb / elapsed:.2f} MB/s)")
    return output_path


def find_matches_per_athlete(video_path, target_names, frame_interval_sec=30, crop_overlay=False):
    """
    Scans video frames and maps detections separately for each athlete.
    Uses cap.grab() for fast-forwarding to avoid keyframe seeking failures.
    """
    print(f"\n[2/4] Scanning Video Frames for Athletes: {target_names}...")
    start_time = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: OpenCV cannot open video file '{video_path}'. Check path and permissions.")
        return {}, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_jump = max(1, int(fps * frame_interval_sec))

    checkpoint = load_checkpoint(video_path, target_names)
    frame_idx = checkpoint.get("last_frame_idx", 0)
    matched_timestamps = checkpoint.get("matched_timestamps", {name: [] for name in target_names})
    frames_checked = checkpoint.get("frames_checked", 0)

    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    if total_frames > 0:
        pbar = tqdm(total=total_frames, initial=frame_idx, unit="frame", desc="Scanning OCR")
    else:
        pbar = tqdm(unit="frame", desc="Scanning OCR")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_checked += 1
        current_sec = frame_idx / fps

        if crop_overlay:
            h, w, _ = frame.shape
            frame = frame[int(h * 0.75):h, 0:w]

        ocr_results = reader.readtext(frame, detail=0)
        detected_text = " ".join(ocr_results).lower()

        for name in target_names:
            if name.lower() in detected_text:
                if name not in matched_timestamps:
                    matched_timestamps[name] = []
                matched_timestamps[name].append(current_sec)
                mins, secs = int(current_sec // 60), int(current_sec % 60)
                pbar.write(f"  ➜ Match detected for '{name}' at {mins:02d}:{secs:02d}")

        save_checkpoint(video_path, frame_idx, matched_timestamps, frames_checked)

        for _ in range(frame_jump - 1):
            if not cap.grab():
                break

        frame_idx += frame_jump
        pbar.update(frame_jump)

    pbar.close()
    cap.release()
    elapsed = time.perf_counter() - start_time
    clear_checkpoint()

    athlete_match_clips = {}
    for name, ts_list in matched_timestamps.items():
        athlete_match_clips[name] = cluster_matches(ts_list)

    return athlete_match_clips, elapsed


def extract_match_clip(input_video, start_sec, end_sec, output_clip):
    """Clips a match segment using FFmpeg stream copy without re-encoding."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", input_video,
        "-c", "copy",
        output_clip
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_clip


def analyze_match_with_gemini(clip_path, athlete_name):
    """Sends clip to Gemini API for technical analysis."""
    print(f"\n[4/4] Analyzing {clip_path} with Gemini AI...")
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    video_file = client.files.upload(file=clip_path)

    prompt = f"""
    You are an expert BJJ black belt coach. Analyze this match clip for athlete: {athlete_name}.
    Provide:
    1. Key Highlights & Timestamps (Takedowns, Guards, Passes, Sweeps, Submissions).
    2. Tactical Breakdown for {athlete_name} (Strengths & mistakes made).
    3. Specific Coaching Recommendations to improve position control or submission defenses.
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[video_file, prompt]
    )
    return response.text


def main():
    parser = argparse.ArgumentParser(description="Automated BJJ Match Finder & Segmenter")
    parser.add_argument("--url", type=str, required=True, help="YouTube URL or local video path")
    parser.add_argument("--names", type=str, nargs="+", required=True, help="Athlete names")
    parser.add_argument("--interval", type=int, default=30, help="Scan interval in seconds (default: 30)")
    parser.add_argument("--crop", action="store_true", help="Enable ROI cropping for bottom 25% overlay")

    args = parser.parse_args()
    total_start = time.perf_counter()

    is_url = args.url.startswith("http://") or args.url.startswith("https://")

    if is_url:
        video_id = extract_youtube_id(args.url)
        target_filename = f"stream_{video_id}.mp4" if video_id else "full_stream.mp4"
        
        # Check if stream already exists locally
        possible_existing = [target_filename, f"{target_filename}.mkv"]
        found_file = next((f for f in possible_existing if os.path.exists(f)), None)

        if found_file:
            print(f"\n[1/4] 💾 Found local video for YouTube ID '{video_id}': '{found_file}'. Skipping download.")
            video_path = found_file
        else:
            video_path = download_stream(args.url, output_path=target_filename)
    else:
        # Local file path provided
        if os.path.exists(args.url):
            print(f"\n[1/4] 📁 Using local video file: '{args.url}'")
            video_path = args.url
        else:
            print(f"\n❌ Error: Local file '{args.url}' was not found.")
            video_files = [f for f in os.listdir(".") if f.endswith((".mp4", ".mkv", ".webm", ".avi"))]
            if video_files:
                print(f"  ➜ Available video files in current directory: {video_files}")
            return

    athlete_matches, ocr_time = find_matches_per_athlete(
        video_path, 
        args.names, 
        frame_interval_sec=args.interval, 
        crop_overlay=args.crop
    )

    print("\n[3/4] Trimming Individual Match Clips...")
    extracted_clips = []

    for name, matches in athlete_matches.items():
        if not matches:
            print(f"❌ No matches found for '{name}'")
            continue

        print(f"\n Found {len(matches)} match(es) for {name}:")
        clean_name = name.replace(" ", "_")

        for idx, (start_sec, end_sec) in enumerate(matches, 1):
            duration = int(end_sec - start_sec)
            output_filename = f"{clean_name}_match_{idx}.mp4"

            extract_match_clip(video_path, start_sec, end_sec, output_filename)
            extracted_clips.append((name, output_filename))

            start_m, start_s = int(start_sec // 60), int(start_sec % 60)
            end_m, end_s = int(end_sec // 60), int(end_sec % 60)
            print(f"  ✓ Saved: {output_filename} ({start_m:02d}:{start_s:02d} to {end_m:02d}:{end_s:02d} | Duration: {duration}s)")

    if os.environ.get("GEMINI_API_KEY") and extracted_clips:
        for athlete_name, clip_file in extracted_clips:
            try:
                breakdown = analyze_match_with_gemini(clip_file, athlete_name)
                print(f"\n" + "="*50)
                print(f"  AI COACH BREAKDOWN: {clip_file}")
                print("="*50)
                print(breakdown)
            except Exception as e:
                print(f"⚠️ Gemini analysis skipped for {clip_file}: {e}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n✅ Pipeline complete in {total_elapsed:.2f}s!")


if __name__ == "__main__":
    main()