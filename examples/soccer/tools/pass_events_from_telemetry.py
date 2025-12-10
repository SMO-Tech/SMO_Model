"""
Infer successful & unsuccessful passes from telemetry logs.

Usage:
    python tools/pass_events_from_telemetry.py \
        --telemetry_path analysis/0bfacc_0/telemetry.jsonl \
        --metadata_path analysis/0bfacc_0/metadata.json

Creates CSV/JSON under analysis/<video>/passes_from_telemetry.*
without touching the real-time demo.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np  # type: ignore[import-untyped]


@dataclass
class Event:
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    passer_id: Optional[int]
    receiver_id: Optional[int]
    passer_team: Optional[int]
    receiver_team: Optional[int]
    distance_m: Optional[float]
    pass_type: str
    outcome: str  # successful | intercepted | lost
    release_speed: Optional[float]
    receive_speed: Optional[float]
    release_recovered: bool
    receive_recovered: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect pass events from telemetry.")
    parser.add_argument("--telemetry_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for output location (defaults to telemetry directory).",
    )
    parser.add_argument("--base_possession_radius_cm", type=float, default=450.0)
    parser.add_argument("--max_dynamic_radius_cm", type=float, default=900.0)
    parser.add_argument("--min_possession_frames", type=int, default=3)
    parser.add_argument("--pass_timeout_frames", type=int, default=35)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    entries: List[dict] = []
    with path.open() as fp:
        for line in fp:
            entries.append(json.loads(line))
    return entries


def nearest_owner(
    ball_pos: np.ndarray,
    players: List[dict],
    radius_cm: float,
) -> Optional[dict]:
    best = None
    best_dist = radius_cm
    for player in players:
        pid = player.get("id")
        pitch = player.get("pitch")
        team = player.get("team")
        if pid is None or pitch is None or team is None:
            continue
        p_vec = np.array(pitch, dtype=float)
        dist = float(np.linalg.norm(p_vec - ball_pos))
        if dist < best_dist:
            best = {"id": pid, "team": team, "dist": dist, "pos": p_vec}
            best_dist = dist
    return best


def classify_distance(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return "unknown"
    if distance_m < 20:
        return "short"
    if distance_m < 35:
        return "medium"
    return "long"


def main() -> None:
    args = parse_args()
    tele_path = Path(args.telemetry_path)
    meta_path = Path(args.metadata_path)
    if not tele_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Telemetry or metadata path not found.")

    telemetry = load_jsonl(tele_path)
    metadata = json.loads(meta_path.read_text())

    output_dir = Path(args.output_dir) if args.output_dir else tele_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "passes_from_telemetry.csv"
    json_path = output_dir / "passes_from_telemetry.json"

    current_owner: Optional[dict] = None
    owner_frames = 0
    owner_last_pos: Optional[np.ndarray] = None
    pending_no_owner = 0

    events: List[Event] = []

    for entry in telemetry:
        frame = entry["frame"]
        timestamp = entry["timestamp"]
        ball_pitch = entry["ball"].get("pitch")
        ball_speed = entry["ball"].get("speed") or 0.0
        ball_recovered = bool(entry["ball"].get("recovered", False))

        dynamic_radius = min(
            args.base_possession_radius_cm + ball_speed * 60.0,
            args.max_dynamic_radius_cm,
        )

        players = entry["players"] + entry.get("goalkeepers", [])

        owner_candidate = None
        owner_id = entry["ball"].get("owner_id")
        owner_team = entry["ball"].get("owner_team")
        if owner_id is not None and owner_team is not None and ball_pitch is not None:
            owner_candidate = {
                "id": owner_id,
                "team": owner_team,
                "pos": np.array(ball_pitch, dtype=float),
                "speed": ball_speed,
                "recovered": ball_recovered,
            }
        elif ball_pitch is not None:
            nearest = nearest_owner(
                np.array(ball_pitch, dtype=float), players, dynamic_radius
            )
            if nearest:
                nearest["speed"] = ball_speed
                nearest["recovered"] = ball_recovered
                owner_candidate = nearest

        if owner_candidate is not None:
            # Ball currently controlled by someone.
            pending_no_owner = 0
            if (
                current_owner is None
                or owner_candidate["id"] != current_owner["id"]
            ):
                # Possession change.
                if (
                    current_owner is not None
                    and owner_frames >= args.min_possession_frames
                ):
                    distance_m = None
                    if owner_last_pos is not None and owner_candidate["pos"] is not None:
                        distance_m = float(
                            np.linalg.norm(owner_candidate["pos"] - owner_last_pos) / 100.0
                        )
                    outcome = (
                        "successful"
                        if owner_candidate["team"] == current_owner["team"]
                        else "intercepted"
                    )
                    events.append(
                        Event(
                            start_frame=current_owner["last_frame"],
                            end_frame=frame,
                            start_time=current_owner["last_time"],
                            end_time=timestamp,
                            passer_id=current_owner["id"],
                            receiver_id=owner_candidate["id"],
                            passer_team=current_owner["team"],
                            receiver_team=owner_candidate["team"],
                            distance_m=distance_m,
                            pass_type=classify_distance(distance_m),
                            outcome=outcome,
                            release_speed=current_owner.get("speed"),
                            receive_speed=owner_candidate.get("speed"),
                            release_recovered=current_owner.get("recovered", False),
                            receive_recovered=owner_candidate.get("recovered", False),
                        )
                    )
                # Reset ownership
                current_owner = {
                    "id": owner_candidate["id"],
                    "team": owner_candidate["team"],
                    "last_frame": frame,
                    "last_time": timestamp,
                    "speed": ball_speed,
                    "recovered": ball_recovered,
                }
                owner_frames = 1
                owner_last_pos = owner_candidate["pos"]
            else:
                # Same owner continues
                owner_frames += 1
                current_owner["last_frame"] = frame
                current_owner["last_time"] = timestamp
                current_owner["speed"] = ball_speed
                current_owner["recovered"] = ball_recovered or current_owner.get("recovered", False)
                owner_last_pos = (
                    owner_candidate["pos"] if owner_candidate["pos"] is not None else owner_last_pos
                )
        else:
            # No owner visible.
            if current_owner is not None:
                pending_no_owner += 1
                if pending_no_owner >= args.pass_timeout_frames:
                    events.append(
                        Event(
                            start_frame=current_owner["last_frame"],
                            end_frame=frame,
                            start_time=current_owner["last_time"],
                            end_time=timestamp,
                            passer_id=current_owner["id"],
                            receiver_id=None,
                            passer_team=current_owner["team"],
                            receiver_team=None,
                            distance_m=None,
                            pass_type="unknown",
                            outcome="lost",
                            release_speed=current_owner.get("speed"),
                            receive_speed=None,
                            release_recovered=current_owner.get("recovered", False),
                            receive_recovered=False,
                        )
                    )
                    current_owner = None
                    owner_frames = 0
                    owner_last_pos = None
            else:
                pending_no_owner = 0

    # Save outputs
    if events:
        with csv_path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "start_time",
                    "end_time",
                    "start_frame",
                    "end_frame",
                    "passer_id",
                    "receiver_id",
                    "passer_team",
                    "receiver_team",
                    "distance_m",
                    "pass_type",
                    "outcome",
                    "release_speed",
                    "receive_speed",
                    "release_recovered",
                    "receive_recovered",
                ]
            )
            for ev in events:
                writer.writerow(
                    [
                        ev.start_time,
                        ev.end_time,
                        ev.start_frame,
                        ev.end_frame,
                        ev.passer_id,
                        ev.receiver_id,
                        ev.passer_team,
                        ev.receiver_team,
                        f"{ev.distance_m:.2f}" if ev.distance_m is not None else "",
                        ev.pass_type,
                        ev.outcome,
                        f"{ev.release_speed:.3f}" if ev.release_speed is not None else "",
                        f"{ev.receive_speed:.3f}" if ev.receive_speed is not None else "",
                        int(ev.release_recovered),
                        int(ev.receive_recovered),
                    ]
                )
        json_path.write_text(json.dumps([ev.__dict__ for ev in events], indent=2))
        print(f"[passes] Detected {len(events)} events -> {csv_path}")
    else:
        print("[passes] No pass events detected.")


if __name__ == "__main__":
    main()


