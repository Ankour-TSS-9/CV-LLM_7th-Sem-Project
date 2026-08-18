"""
event_extractor.py

Walks the `tracks` dict already produced by trackers.Tracker / your existing
pipeline (tracks['players'], tracks['ball'], tracks['referees']) and turns it
into a compact, timestamped list of match "events". This event list is what
gets sent to the LLM for commentary generation -- much cheaper and more
reliable than feeding raw per-frame data to an LLM.

Expects each player track entry to look like (matching your existing repo):
    tracks['players'][frame_num][track_id] = {
        'bbox': [...],
        'team': 1 or 2,
        'team_color': (b, g, r),
        'has_ball': bool,          # set by PlayerBallAssigner in main.py
        'speed': float,            # km/h, set by SpeedAndDistance_Estimator
        'distance': float,         # meters, cumulative
        'position_transformed': [x, y] or None,
    }

If your `has_ball` flag isn't already being set on every frame (it's only
set on frames where PlayerBallAssigner ran in the original repo), call
`main.py`'s existing loop first so tracks['players'][...]['has_ball'] exists
before calling extract_events.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchEvent:
    time: float          # seconds into the clip
    frame: int
    type: str            # "possession_change" | "sprint" | "ball_speed"
    team: Optional[int] = None
    player_id: Optional[int] = None
    detail: dict = field(default_factory=dict)

    def to_dict(self):
        d = {"time": round(float(self.time), 2), "frame": int(self.frame), "type": self.type}
        if self.team is not None:
            d["team"] = int(self.team)
        if self.player_id is not None:
            d["player_id"] = int(self.player_id)
        # cast any numpy scalar types in detail (e.g. speed_kmh) to native Python types
        clean_detail = {}
        for k, v in self.detail.items():
            if hasattr(v, "item"):  # numpy scalar
                v = v.item()
            clean_detail[k] = v
        d.update(clean_detail)
        return d


def extract_events(
    tracks: dict,
    fps: int = 24,
    sprint_kmh_threshold: float = 22.0,
    min_frames_between_same_sprint: int = 30,
) -> list[dict]:
    """
    Returns a list of event dicts, sorted by time, e.g.:
        {"time": 4.2, "frame": 101, "type": "possession_change",
         "team": 1, "player_id": 7}
        {"time": 6.0, "frame": 144, "type": "sprint",
         "team": 2, "player_id": 14, "speed_kmh": 27.3}

    This is intentionally conservative -- it only emits events with a clear
    signal, so the LLM isn't drowning in noise from tracking jitter.
    """
    events: list[MatchEvent] = []

    player_frames = tracks.get("players", [])
    num_frames = len(player_frames)

    # --- Possession changes -------------------------------------------------
    last_holder = None  # (team, player_id)
    for frame_num in range(num_frames):
        frame_players = player_frames[frame_num]
        holder = None
        for player_id, info in frame_players.items():
            if info.get("has_ball"):
                holder = (info.get("team"), player_id)
                break

        if holder is not None and holder != last_holder:
            events.append(
                MatchEvent(
                    time=frame_num / fps,
                    frame=frame_num,
                    type="possession_change",
                    team=holder[0],
                    player_id=holder[1],
                )
            )
            last_holder = holder

    # --- Sprints --------------------------------------------------------------
    last_sprint_frame_by_player: dict[int, int] = {}
    for frame_num in range(num_frames):
        frame_players = player_frames[frame_num]
        for player_id, info in frame_players.items():
            speed = info.get("speed")
            if speed is None or speed < sprint_kmh_threshold:
                continue
            last_frame = last_sprint_frame_by_player.get(player_id, -10_000)
            if frame_num - last_frame < min_frames_between_same_sprint:
                continue  # don't spam repeated events for the same sprint
            last_sprint_frame_by_player[player_id] = frame_num
            events.append(
                MatchEvent(
                    time=frame_num / fps,
                    frame=frame_num,
                    type="sprint",
                    team=info.get("team"),
                    player_id=player_id,
                    detail={"speed_kmh": round(speed, 1)},
                )
            )

    # --- Ball speed bursts (possible pass / shot) ------------------------------
    ball_frames = tracks.get("ball", [])
    prev_pos = None
    prev_frame = None
    for frame_num, frame_ball in enumerate(ball_frames):
        if 1 not in frame_ball:
            continue
        pos = frame_ball[1].get("position_transformed")
        if pos is None:
            continue
        if prev_pos is not None and prev_frame is not None:
            dt = (frame_num - prev_frame) / fps
            if dt > 0:
                dx = pos[0] - prev_pos[0]
                dy = pos[1] - prev_pos[1]
                dist = (dx**2 + dy**2) ** 0.5
                speed_mps = dist / dt
                speed_kmh = speed_mps * 3.6
                if speed_kmh > 40:  # fast ball movement -> likely a pass/shot
                    events.append(
                        MatchEvent(
                            time=frame_num / fps,
                            frame=frame_num,
                            type="ball_speed",
                            detail={"speed_kmh": round(speed_kmh, 1)},
                        )
                    )
        prev_pos = pos
        prev_frame = frame_num

    events.sort(key=lambda e: e.time)
    return [e.to_dict() for e in events]


if __name__ == "__main__":
    import pickle
    import json
    import sys

    stub_path = sys.argv[1] if len(sys.argv) > 1 else "stubs/track_stubs.pkl"
    with open(stub_path, "rb") as f:
        tracks = pickle.load(f)

    evts = extract_events(tracks)
    print(json.dumps(evts, indent=2))