"""
tts_generator.py  (ElevenLabs version)

Converts each timestamped commentary line into an mp3 clip using ElevenLabs.

Requires:
    pip install elevenlabs

Set your API key as an environment variable before running:
    $env:ELEVENLABS_API_KEY="your-key-here"     (current PowerShell session)
    setx ELEVENLABS_API_KEY "your-key-here"     (Windows, permanent)

Get a key at https://elevenlabs.io/app/settings/api-keys

Pick a voice: run this file with --list-voices to print voice names + IDs
from your account, then set VOICE_ID below to the one you want. A deep,
energetic male voice (e.g. "Adam" or "Josh") tends to suit sports commentary
well -- but try a few, it's personal preference.
"""

import os

from elevenlabs import ElevenLabs, VoiceSettings

# Default ElevenLabs premade voice: "Adam" -- swap this for any voice_id
# from your account (custom or premade). Run with --list-voices to see options.
VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam
MODEL_ID = "eleven_turbo_v2_5"  # fast + cheap; use "eleven_multilingual_v2" for higher quality

# Tuned for excited, dynamic broadcast delivery rather than flat/calm reading.
VOICE_SETTINGS = VoiceSettings(
    stability=0.35,          # lower = more expressive/varied delivery
    similarity_boost=0.75,
    style=0.65,               # higher = more exaggerated emotion
    use_speaker_boost=True,
)


def _client() -> ElevenLabs:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY environment variable not set. "
            "Get a key at https://elevenlabs.io/app/settings/api-keys"
        )
    return ElevenLabs(api_key=api_key)


def generate_audio_clips(
    commentary: list[dict],
    output_dir: str,
    voice_id: str = VOICE_ID,
) -> list[dict]:
    """
    commentary: [{"time": 1.2, "text": "..."}]
    Returns:    [{"time": 1.2, "text": "...", "audio_path": "commentary_audio/line_000.mp3"}]
    """
    os.makedirs(output_dir, exist_ok=True)
    client = _client()
    results = []

    for i, line in enumerate(commentary):
        out_path = os.path.join(output_dir, f"line_{i:03d}.mp3")

        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=MODEL_ID,
            text=line["text"],
            voice_settings=VOICE_SETTINGS,
        )

        with open(out_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        results.append({**line, "audio_path": out_path})

    return results


def list_voices():
    client = _client()
    voices = client.voices.get_all()
    for v in voices.voices:
        print(f"{v.name:20s}  {v.voice_id}")


if __name__ == "__main__":
    import json
    import sys

    if "--list-voices" in sys.argv:
        list_voices()
        sys.exit(0)

    commentary_path = sys.argv[1] if len(sys.argv) > 1 else "commentary_lines.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "commentary_audio"

    with open(commentary_path) as f:
        commentary = json.load(f)

    clips = generate_audio_clips(commentary, out_dir)
    print(json.dumps(clips, indent=2))