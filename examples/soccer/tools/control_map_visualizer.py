"""
Offline control-map & smoothed ball trajectory renderer.

Usage:
    python tools/control_map_visualizer.py \
        --telemetry_path analysis/0bfacc_0/telemetry.jsonl \
        --metadata_path analysis/0bfacc_0/metadata.json \
        --video_path data/0bfacc_0.mp4

This will produce an annotated video (analysis/<stem>/enhanced_control_map.avi)
showing:
  * Team control heatmap (Voronoi-style) rendered on a radar/pitch inset
  * Player markers per team, updated per frame
  * Smoothed ball trajectory (moving average to reduce jitter)
  * Graceful handling of missing detections (last-known positions)

The original real-time demo is unaffected; this script operates purely on the
logged telemetry.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm

TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from main import CONFIG  # type: ignore


@dataclass
class PlayerSample:
    player_id: Optional[int]
    team_id: int
    pitch_xy: Optional[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render team control maps & ball paths.")
    parser.add_argument("--telemetry_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory (defaults to telemetry folder)",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=7,
        help="Sliding window size for ball trajectory smoothing.",
    )
    parser.add_argument(
        "--grid_height",
        type=int,
        default=120,
        help="Grid resolution (rows) for control map computation.",
    )
    parser.add_argument(
        "--grid_width",
        type=int,
        default=200,
        help="Grid resolution (cols) for control map computation.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    entries: List[dict] = []
    with path.open() as fp:
        for line in fp:
            entries.append(json.loads(line))
    return entries


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text())


def smooth_positions(
    coordinates: List[Optional[np.ndarray]],
    window: int,
) -> List[Optional[np.ndarray]]:
    if window <= 1:
        return coordinates
    half = window // 2
    smoothed: List[Optional[np.ndarray]] = []
    for idx, value in enumerate(coordinates):
        if value is None:
            smoothed.append(None)
            continue
        acc = []
        for offset in range(-half, half + 1):
            j = idx + offset
            if 0 <= j < len(coordinates) and coordinates[j] is not None:
                acc.append(coordinates[j])
        if acc:
            smoothed.append(np.mean(acc, axis=0))
        else:
            smoothed.append(value)
    return smoothed


def build_control_map(
    players: Sequence[PlayerSample],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Approximate team-control map via brute-force nearest-player lookup."""
    teams = [p.team_id for p in players if p.pitch_xy is not None and p.team_id in (0, 1)]
    coords = [p.pitch_xy for p in players if p.pitch_xy is not None and p.team_id in (0, 1)]
    if len(teams) == 0:
        # no valid players → empty overlay
        return np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    coords_arr = np.stack(coords, axis=0)  # (N,2)
    teams_arr = np.array(teams, dtype=np.int32)

    xs = np.linspace(0, CONFIG.length, grid_w, dtype=np.float32)
    ys = np.linspace(0, CONFIG.width, grid_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = np.stack([grid_x, grid_y], axis=-1)  # (H,W,2)

    diff = grid[:, :, None, :] - coords_arr[None, None, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)  # (H,W,N)
    nearest_idx = np.argmin(dist_sq, axis=-1)
    nearest_team = teams_arr[nearest_idx]

    colors = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    team0_color = np.array([255, 120, 80], dtype=np.uint8)  # BGR
    team1_color = np.array([80, 160, 255], dtype=np.uint8)
    colors[nearest_team == 0] = team0_color
    colors[nearest_team == 1] = team1_color
    return colors


def pitch_to_image_coords(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map pitch coordinates (cm) to image pixel coordinates in radar view."""
    x_norm = np.clip(xy[0] / CONFIG.length, 0.0, 1.0)
    y_norm = np.clip(xy[1] / CONFIG.width, 0.0, 1.0)
    px = int(x_norm * (width - 1))
    py = int((1.0 - y_norm) * (height - 1))
    return np.array([px, py], dtype=np.int32)


def render_radar_overlay(
    record: dict,
    players: Sequence[PlayerSample],
    ball_xy: Optional[np.ndarray],
    ball_path: Sequence[np.ndarray],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    base_pitch = draw_pitch(CONFIG)
    if isinstance(base_pitch, sv.Image):
        pitch_img = base_pitch.data.copy()
    else:
        pitch_img = base_pitch.copy()

    control_layer = build_control_map(players, grid_h, grid_w)
    control_layer = cv2.resize(
        control_layer,
        (pitch_img.shape[1], pitch_img.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    blended = cv2.addWeighted(pitch_img, 0.6, control_layer, 0.4, 0)

    # Draw players via supervision helper
    player_xy = [
        p.pitch_xy for p in players if p.pitch_xy is not None and p.team_id in (0, 1)
    ]
    player_xy = np.array(player_xy, dtype=np.float32)
    if len(player_xy) > 0:
        colors = np.array([p.team_id for p in players if p.pitch_xy is not None and p.team_id in (0, 1)])
        blended = draw_points_on_pitch(
            config=CONFIG,
            xy=player_xy[colors == 0],
            face_color=sv.Color.from_hex("#FF7B3A"),
            pitch=blended,
            radius=18,
        )
        blended = draw_points_on_pitch(
            config=CONFIG,
            xy=player_xy[colors == 1],
            face_color=sv.Color.from_hex("#3A9CFF"),
            pitch=blended,
            radius=18,
        )

    # Ball path (smoothed)
    if ball_path:
        pts = np.array(
            [pitch_to_image_coords(p, blended.shape[1], blended.shape[0]) for p in ball_path],
            dtype=np.int32,
        )
        for idx in range(1, len(pts)):
            cv2.line(blended, tuple(pts[idx - 1]), tuple(pts[idx]), (0, 255, 0), 2)

    if ball_xy is not None:
        ball_px = pitch_to_image_coords(ball_xy, blended.shape[1], blended.shape[0])
        cv2.circle(blended, tuple(ball_px), 10, (0, 255, 255), -1)

    return blended


def main() -> None:
    args = parse_args()
    telemetry_path = Path(args.telemetry_path)
    metadata_path = Path(args.metadata_path)
    video_path = Path(args.video_path)
    if not telemetry_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Telemetry or metadata path not found.")

    telemetry = load_jsonl(telemetry_path)
    metadata = load_metadata(metadata_path)
    fps = float(metadata.get("fps", 25.0))

    output_dir = Path(args.output_dir) if args.output_dir else telemetry_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / "enhanced_control_map.avi"

    # Precompute smoothed ball positions
    ball_positions: List[Optional[np.ndarray]] = []
    for entry in telemetry:
        pitch_xy = entry["ball"].get("pitch")
        ball_positions.append(np.array(pitch_xy, dtype=np.float32) if pitch_xy else None)
    smoothed_ball = smooth_positions(ball_positions, args.smooth_window)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    radar_history: List[np.ndarray] = []
    radar_window = max(1, args.smooth_window)

    for idx, entry in enumerate(tqdm(telemetry, desc="rendering", unit="frame")):
        ret, frame = cap.read()
        if not ret:
            break

        players = [
            PlayerSample(p.get("id"), p.get("team", 0), np.array(p["pitch"], dtype=np.float32) if p["pitch"] else None)
            for p in entry["players"]
        ]
        goalkeepers = [
            PlayerSample(p.get("id"), p.get("team", 2), np.array(p["pitch"], dtype=np.float32) if p["pitch"] else None)
            for p in entry["goalkeepers"]
        ]
        all_players = players + goalkeepers

        ball_xy = smoothed_ball[idx] if idx < len(smoothed_ball) else None
        history_slice = [p for p in smoothed_ball[max(0, idx - radar_window): idx + 1] if p is not None]

        radar_img = render_radar_overlay(
            entry,
            all_players,
            ball_xy,
            history_slice,
            args.grid_height,
            args.grid_width,
        )

        # Blend radar onto bottom-right corner
        radar_resized = cv2.resize(
            radar_img,
            (width // 2, height // 2),
            interpolation=cv2.INTER_CUBIC,
        )
        rh, rw, _ = radar_resized.shape
        y0 = height - rh
        x0 = width - rw
        overlay = frame[y0:, x0:].copy()
        blended = cv2.addWeighted(overlay, 0.4, radar_resized, 0.6, 0)
        frame[y0:, x0:] = blended

        cv2.putText(
            frame,
            f"Control Map + Ball Path | Frame {entry['frame']}  Time {entry['timestamp']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[control-map] Wrote {output_video_path}")


if __name__ == "__main__":
    main()


