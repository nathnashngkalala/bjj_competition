import argparse
import time
import os
import cv2
import easyocr
import subprocess
from google import genai

# Initialize EasyOCR reader (runs locally)
reader = easyocr.Reader(['en'], gpu=False)


def download_stream(youtube_url, output_path="full_stream.mp4"):
    """Downloads the YouTube stream via yt-dlp and benchmarks download time."""
    print(f"\n[1/4] Downloading YouTube Stream...")
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
    return output_path, elapsed


def find_match_timestamps(video_path, target_names, frame_interval_sec=5, crop_overlay=False):
    """
    Scans video frames using OCR to detect athlete names.
    Includes benchmarking for OCR duration and processing speed.
    """
    print(f"\n[2/4] Scanning Video Frames for Athletes: {target_names}...")
    start_time = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_jump = max(1, int(fps * frame_interval_sec))

    matched_timestamps = []
    frames_checked = 0
    frame_idx = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frames_checked += 1
        current_sec = frame_idx / fps

        # Optional Optimization: Crop to top/bottom overlay bar where names usually appear
        if crop_overlay:
            h, w, _ = frame.shape
            # Focus on lower 25% of the screen (adjust as needed for Scrambleclash overlays)
            frame = frame[int(h * 0.75):h, 0:w]

        # Run local OCR
        ocr_results = reader.readtext(frame, detail=0)
        detected_text = " ".join(ocr_results).lower()

        for name in target_names:
            if name.lower() in detected_text:
                matched_timestamps.append(current_sec)
                mins, secs = int(current_sec // 60), int(current_sec % 60)
                print(f"  ➜ Match detected for '{name}' at {mins:02d}:{secs:02d}")

        frame_idx += frame_jump

    cap.release()
    elapsed = time.perf_counter() - start_time

    # Performance Stats
    fps_processed = frames_checked / elapsed if elapsed > 0 else 0
    print(f"✓ OCR Scan Complete:")
    print(f"  • Total Video Duration: {int(total_frames / fps // 60)}m {int(total_frames / fps % 60)}s")
    print(f"  • Frames Inspected: {frames_checked}")
    print(f"  • OCR Processing Speed: {fps_processed:.2f} frames/sec")
    print(f"  • Time Taken: {elapsed:.2f}s")

    if not matched_timestamps:
        return None, None, elapsed

    start_sec = max(0, min(matched_timestamps) - 30)
    end_sec = max(matched_timestamps) + 30
    return start_sec, end_sec, elapsed


def extract_match_clip(input_video, start_sec, end_sec, output_clip="match_clip.mp4"):
    """Clips the detected match segment without re-encoding."""
    print(f"\n[3/4] Trimming Match Clip ({int(start_sec)}s to {int(end_sec)}s)...")
    start_time = time.perf_counter()

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", input_video,
        "-c", "copy",
        output_clip
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    elapsed = time.perf_counter() - start_time
    print(f"✓ Clip Extracted in {elapsed:.2f}s -> {output_clip}")
    return output_clip, elapsed


def analyze_match_with_gemini(clip_path, target_names):
    """Sends clip to Gemini API for multi-athlete technical analysis."""
    print(f"\n[4/4] Sending Clip to Gemini API for Technical Analysis...")
    start_time = time.perf_counter()

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    video_file = client.files.upload(file=clip_path)

    athletes_str = ", ".join(target_names)
    prompt = f"""
    You are an expert BJJ black belt coach analyzing a tournament video clip.
    Analyze the match featuring athlete(s): {athletes_str}.

    Provide:
    1. Key Match Highlights & Timestamps (Takedowns, Guards, Passes, Sweeps, Submissions).
    2. Tactical Breakdown for {athletes_str} (Strengths & mistakes made).
    3. Specific Coaching Recommendations to improve position control or submission defenses.
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[video_file, prompt]
    )

    elapsed = time.perf_counter() - start_time
    print(f"✓ Gemini Analysis Completed in {elapsed:.2f}s")
    return response.text, elapsed


def main():
    parser = argparse.ArgumentParser(description="Automated BJJ Match Finder & Performance Benchmarker")
    parser.add_argument("--url", type=str, required=True, help="YouTube livestream video URL")
    parser.add_argument("--names", type=str, nargs="+", required=True, help="List of athlete names to find (space-separated)")
    parser.add_argument("--interval", type=int, default=5, help="Scan interval in seconds (default: 5)")
    parser.add_argument("--crop", action="store_true", help="Enable ROI cropping for faster OCR")

    args = parser.parse_args()

    total_start = time.perf_counter()

    # Pipeline execution
    video_path, dl_time = download_stream(args.url)
    start_sec, end_sec, ocr_time = find_match_timestamps(
        video_path, 
        args.names, 
        frame_interval_sec=args.interval, 
        crop_overlay=args.crop
    )

    if start_sec is not None and end_sec is not None:
        clip_path, clip_time = extract_match_clip(video_path, start_sec, end_sec)
        breakdown, api_time = analyze_match_with_gemini(clip_path, args.names)

        total_elapsed = time.perf_counter() - total_start

        print("\n" + "="*50)
        print("          AI COACH TECHNICAL BREAKDOWN          ")
        print("="*50)
        print(breakdown)

        print("\n" + "="*50)
        print("             PERFORMANCE BENCHMARKS             ")
        print("="*50)
        print(f"• Stream Download Time:  {dl_time:.2f}s")
        print(f"• OCR Search Time:      {ocr_time:.2f}s")
        print(f"• FFmpeg Clipping Time: {clip_time:.2f}s")
        print(f"• Gemini API Analysis:  {api_time:.2f}s")
        print(f"• Total Pipeline Time:  {total_elapsed:.2f}s")
        print("="*50)
    else:
        print("\n❌ Could not locate specified athletes in this stream.")


if __name__ == "__main__":
    main()