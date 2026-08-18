"""
audio_merger.py

Takes the annotated output video (from your existing CV pipeline, e.g.
output_videos/output_video.avi) and the list of timestamped audio clips
from tts_generator.py, and produces a single video with all the commentary
lines mixed in at the right times.

Requires ffmpeg installed and on PATH (you already have it -- you used it
earlier to trim clips).

Note: your existing output_video.avi likely has no audio track (the CV
pipeline only draws overlays on frames). This script adds one from scratch.
If your source video *does* have original crowd/match audio you want kept
underneath the commentary, see the `keep_original_audio` flag.
"""

import os
import subprocess
import json


def _get_audio_duration(path: str) -> float:
    """Uses ffprobe (bundled with ffmpeg) to get a clip's duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", path,
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def resolve_overlaps(audio_clips: list[dict], min_gap: float = 0.15) -> list[dict]:
    """
    Commentary timestamps come from the LLM, which doesn't know how long each
    line will actually take to *say*. If line N+1's timestamp falls before
    line N finishes speaking, they'd play simultaneously (sounds like two
    commentators talking over each other). This pushes any overlapping line
    forward so it starts only after the previous one ends, with a small gap.
    """
    clips = sorted(audio_clips, key=lambda c: c["time"])
    adjusted = []
    prev_end = -1e9

    for clip in clips:
        duration = _get_audio_duration(clip["audio_path"])
        start = max(clip["time"], prev_end + min_gap)
        adjusted.append({**clip, "time": start, "_duration": duration})
        prev_end = start + duration

    return adjusted


def merge_commentary_into_video(
    video_path: str,
    audio_clips: list[dict],
    output_path: str,
    original_audio_path: str | None = None,
    original_audio_volume: float = 0.15,
) -> str:
    """
    audio_clips: [{"time": 1.2, "audio_path": "commentary_audio/line_000.mp3"}, ...]

    If original_audio_path is given (e.g. the source clip before annotation,
    which still has real match/crowd audio), it's mixed in underneath the
    commentary at `original_audio_volume` (0.0-1.0). Otherwise the output
    has commentary only.
    """
    if not audio_clips:
        raise ValueError("No audio clips to merge -- did commentary generation run?")

    audio_clips = resolve_overlaps(audio_clips)

    # Build ffmpeg filter_complex: delay each clip to its timestamp, then mix all.
    inputs = ["-i", video_path]
    filter_parts = []
    mix_inputs = []

    input_index = 1  # 0 is the video
    for clip in audio_clips:
        inputs += ["-i", clip["audio_path"]]
        delay_ms = int(clip["time"] * 1000)
        label = f"a{input_index}"
        filter_parts.append(
            f"[{input_index}:a]adelay={delay_ms}|{delay_ms}[{label}]"
        )
        mix_inputs.append(f"[{label}]")
        input_index += 1

    if original_audio_path:
        inputs += ["-i", original_audio_path]
        filter_parts.append(
            f"[{input_index}:a]volume={original_audio_volume}[orig]"
        )
        mix_inputs.append("[orig]")
        input_index += 1

    filter_complex = ";".join(filter_parts)
    filter_complex += (
        f";{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]

    subprocess.run(cmd, check=True)
    return output_path


if __name__ == "__main__":
    import json
    import sys

    video_path = sys.argv[1]
    clips_json_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output_videos/output_with_commentary.mp4"

    with open(clips_json_path) as f:
        clips = json.load(f)

    merge_commentary_into_video(video_path, clips, output_path)
    print(f"Wrote {output_path}")