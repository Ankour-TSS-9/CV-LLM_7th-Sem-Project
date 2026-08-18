"""
run_commentary.py

The orchestrator -- run this after your existing main.py finishes (or add
its logic to the end of main.py, see integration notes below).

Usage:
    python commentary/run_commentary.py

Assumes (matching your repo's current layout):
    - stubs/track_stubs.pkl              already written by main.py
    - output_videos/output_video.avi     already written by main.py
    - fps = 24 (change FPS below if different)

Produces:
    - commentary/commentary_events.json  (debug: the extracted events)
    - commentary/commentary_lines.json   (debug: what Claude wrote)
    - commentary_audio/*.mp3             (debug: individual voice clips)
    - output_videos/output_with_commentary.mp4   <- final deliverable
"""

import json
import pickle

from event_extractor import extract_events
from commentary_generator import generate_commentary
from tts_generator import generate_audio_clips
from audio_merger import merge_commentary_into_video

FPS = 24
STUB_PATH = "stubs/full_tracks.pkl"
VIDEO_PATH = "output_videos/output_video.avi"
FINAL_OUTPUT_PATH = "output_videos/output_with_commentary.mp4"


def main():
    print("Loading tracks...")
    with open(STUB_PATH, "rb") as f:
        tracks = pickle.load(f)

    num_frames = len(tracks.get("players", []))
    duration_seconds = num_frames / FPS

    print(f"Extracting events from {num_frames} frames ({duration_seconds:.1f}s)...")
    events = extract_events(tracks, fps=FPS)
    with open("commentary/commentary_events.json", "w") as f:
        json.dump(events, f, indent=2)
    print(f"Found {len(events)} events.")

    print("Generating commentary with Claude...")
    lines = generate_commentary(events, clip_duration_seconds=duration_seconds)
    with open("commentary/commentary_lines.json", "w") as f:
        json.dump(lines, f, indent=2)
    print(f"Got {len(lines)} commentary lines.")

    print("Synthesizing speech (Edge-TTS)...")
    clips = generate_audio_clips(lines, output_dir="commentary_audio")

    print("Merging commentary into video...")
    merge_commentary_into_video(VIDEO_PATH, clips, FINAL_OUTPUT_PATH)

    print(f"\nDone! Final video: {FINAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()