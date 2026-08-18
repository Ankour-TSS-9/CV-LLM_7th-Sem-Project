"""
commentary_generator.py  (Gemini free-tier version)

Takes the event timeline from event_extractor.py and asks Gemini to write
excited sports-broadcast-style commentary lines, each tagged with a
timestamp so tts_generator.py knows exactly when to play each line.

Requires:
    pip install google-genai

Set your API key as an environment variable before running:
    $env:GEMINI_API_KEY="your-key-here"      (current PowerShell session)
    setx GEMINI_API_KEY "your-key-here"      (Windows, permanent)

Get a free key at https://aistudio.google.com/apikey
"""

import json
import os
from google import genai

SYSTEM_PROMPT = """\
You are an excited, energetic football (soccer) TV commentator. You will be \
given a JSON list of match events with timestamps (in seconds) extracted \
from computer-vision tracking of a short clip. Write commentary lines that \
sound like a real live broadcast -- fast-paced, enthusiastic, natural \
phrasing, occasional filler ("Oh!", "And there it is!", "Look at that pace!").

Rules:
- Output ONLY a JSON array, no other text, no markdown code fences.
- Each item: {"time": <seconds, float>, "text": "<commentary line>"}
- Use the given event timestamps as a guide but you may shift a line by up \
to 0.5s for natural phrasing.
- Don't narrate every single event robotically -- combine nearby events into \
one natural sentence where it makes sense (e.g. a possession change right \
before a sprint can be one line).
- Keep each line short enough to say in 2-4 seconds (roughly 8-15 words).
- Refer to players by their track ID number (e.g. "number 7") since that's \
all the identifying info available -- don't invent player names.
- If two events are very close in time (<1s apart), you may merge them or \
only comment on the more exciting one -- don't cram everything in.
- Leave natural gaps of silence where nothing notable is happening -- not \
every second of the clip needs a line.
"""


def generate_commentary(
    events: list[dict],
    clip_duration_seconds: float,
    model: str = "gemini-3.6-flash",
    api_key: str | None = None,
) -> list[dict]:
    """
    Returns a list like:
        [{"time": 1.2, "text": "And City win it back straight away!"},
         {"time": 4.8, "text": "Number 14 bursts forward with real pace!"}]
    """
    client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    user_content = json.dumps(
        {
            "clip_duration_seconds": round(clip_duration_seconds, 2),
            "events": events,
        }
    )

    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
        },
    )

    raw_text = response.text.strip()

    # Gemini usually respects response_mime_type=json, but strip fences just in case.
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    lines = json.loads(raw_text)
    lines.sort(key=lambda x: x["time"])
    return lines


if __name__ == "__main__":
    import sys

    events_path = sys.argv[1] if len(sys.argv) > 1 else "commentary_events.json"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    with open(events_path) as f:
        events = json.load(f)

    lines = generate_commentary(events, duration)
    print(json.dumps(lines, indent=2))