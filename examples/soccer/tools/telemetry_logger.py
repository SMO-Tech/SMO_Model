"""
Telemetry logger for the Soccer demo.

This script reuses the existing YOLO + tracking pipeline to capture per-frame
player, goalkeeper, referee, and ball metadata (image coordinates, pitch
coordinates, tracker IDs, team assignments, etc). Output is written to
analysis/<video_stem>/telemetry.jsonl plus a metadata.json with video details.

The original demo code under main.py is left untouched; this script can be
executed independently:

    python tools/telemetry_logger.py --source_video_path data/0bfacc_0.mp4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytesseract
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# Allow "import main" and sports.* modules without modifying sys.path globally
TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from sports.common.ball import BallTracker
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from main import (  # type: ignore
    BALL_CLASS_ID,
    BALL_DETECTION_MODEL_PATH,
    CONFIG,
    ELLIPSE_ANNOTATOR,
    ELLIPSE_LABEL_ANNOTATOR,
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    PLAYER_DETECTION_MODEL_PATH,
    PITCH_DETECTION_MODEL_PATH,
    REFEREE_CLASS_ID,
    STRIDE,
    TRIANGLE_ANNOTATOR,
    get_crops,
    resolve_goalkeepers_team_id,
)
from tools.lineup_loader import RosterEntry, load_lineup_csv, roster_to_lookup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log per-frame telemetry.")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Optional custom output directory. Default: analysis/<video_stem>",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, mps…")
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=STRIDE,
        help="Stride for collecting player crops (team classifier fitting).",
    )
    parser.add_argument("--home_lineup_path", type=str, default=None, help="CSV from lineup extractor for the home team.")
    parser.add_argument("--away_lineup_path", type=str, default=None, help="CSV from lineup extractor for the away team.")
    parser.add_argument("--home_team_name", type=str, default="Home", help="Name used for home team metadata.")
    parser.add_argument("--away_team_name", type=str, default="Away", help="Name used for away team metadata.")
    return parser.parse_args()


def ensure_output_paths(video_path: Path, output_dir: Optional[Path]) -> Dict[str, Path]:
    if output_dir is None:
        base_dir = video_path.parent / "analysis" / video_path.stem
    else:
        base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = base_dir / "frames_with_detections"
    frames_dir.mkdir(exist_ok=True)
    missing_ball_dir = base_dir / "frames_missing_ball"
    missing_ball_dir.mkdir(exist_ok=True)
    return {
        "base": base_dir,
        "telemetry": base_dir / "telemetry.jsonl",
        "metadata": base_dir / "metadata.json",
        "gaps": base_dir / "ball_gap_windows.json",
        "frames_dir": frames_dir,
        "missing_ball_dir": missing_ball_dir,
    }


def load_models(device: str) -> Dict[str, YOLO]:
    print("[telemetry] Loading models…")
    models = {
        "player": YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device),
        "pitch": YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device),
        "ball": YOLO(BALL_DETECTION_MODEL_PATH).to(device=device),
    }
    return models


def collect_player_crops(
    player_model: YOLO,
    video_path: Path,
    stride: int,
) -> List[np.ndarray]:
    print(f"[telemetry] Collecting player crops (stride={stride})…")
    crops: List[np.ndarray] = []
    frame_gen = sv.get_video_frames_generator(str(video_path), stride=stride)
    for frame in tqdm(frame_gen, desc="collecting", unit="frame"):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops.extend(get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID]))
        if len(crops) >= 200:
            break
    return crops


def fit_team_classifier(device: str, crops: List[np.ndarray]) -> Optional[TeamClassifier]:
    if len(crops) < 2:
        print("[telemetry] Warning: insufficient crops to fit TeamClassifier.")
        return None
    classifier = TeamClassifier(device=device)
    classifier.fit(crops)
    return classifier


def nearest_owner(
    ball_pitch: np.ndarray,
    players: List[dict],
    radius_cm: float,
) -> Optional[dict]:
    candidate = None
    best_dist = radius_cm
    for entity in players:
        pid = entity.get("id")
        team = entity.get("team")
        pitch = entity.get("pitch")
        if pid is None or team is None or pitch is None:
            continue
        player_vec = np.array(pitch, dtype=float)
        dist = float(np.linalg.norm(player_vec - ball_pitch))
        if dist < best_dist:
            candidate = {
                "id": pid,
                "team": team,
                "dist": dist,
                "pos": player_vec,
            }
            best_dist = dist
    return candidate


def serialize_entities(
    det: sv.Detections,
    teams: np.ndarray,
    transformer: Optional[ViewTransformer],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    if len(det) == 0:
        return result
    anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(np.float32)
    pitch_pts = transformer.transform_points(anchors) if transformer is not None else None
    for idx, tid in enumerate(det.tracker_id):
        entry = {
            "id": int(tid) if tid is not None else None,
            "team": int(teams[idx]) if len(teams) > idx else 0,
            "confidence": float(det.confidence[idx]) if det.confidence is not None else None,
            "bbox": det.xyxy[idx].astype(float).tolist(),
            "image": anchors[idx].astype(float).tolist(),
            "pitch": pitch_pts[idx].astype(float).tolist() if pitch_pts is not None else None,
        }
        result.append(entry)
    return result


def extract_jersey_digits(crop: np.ndarray) -> Optional[str]:
    if crop.size == 0:
        return None
    h = crop.shape[0]
    if h < 10:
        return None
    focus = crop[int(h * 0.3) :, :]
    gray = cv2.cvtColor(focus, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    config = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    digits = re.sub(r"[^0-9]", "", text)
    if 1 <= len(digits) <= 2:
        return digits
    return None


def update_jersey_memory(
    detections: sv.Detections,
    crops: List[np.ndarray],
    memory: Dict[int, str],
) -> None:
    if len(detections) == 0 or detections.tracker_id is None:
        return
    for idx, crop in enumerate(crops):
        if idx >= len(detections):
            break
        tracker_id = detections.tracker_id[idx]
        if tracker_id is None or tracker_id in memory:
            continue
        jersey = extract_jersey_digits(crop)
        if jersey:
            memory[int(tracker_id)] = jersey


def enrich_entities_with_roster(
    entities: List[Dict[str, Any]],
    jersey_memory: Dict[int, str],
    rosters: Optional[Dict[int, Dict[str, Dict[str, Optional[str]]]]],
) -> None:
    if not entities:
        return
    for entry in entities:
        tracker_id = entry.get("id")
        if tracker_id is None:
            continue
        jersey = jersey_memory.get(int(tracker_id))
        if jersey:
            entry["jersey_number"] = jersey
            if rosters:
                team = entry.get("team")
                if team is not None:
                    team_lookup = rosters.get(int(team))
                    if team_lookup:
                        roster_hit = team_lookup.get(jersey)
                        if roster_hit:
                            entry["player_name"] = roster_hit.get("player_name")
                            entry["team_name"] = roster_hit.get("team_name")


def write_metadata(path: Path, info: Dict[str, Any]) -> None:
    path.write_text(json.dumps(info, indent=2))
    print(f"[telemetry] Metadata -> {path}")


def log_frames(
    models: Dict[str, YOLO],
    video_path: Path,
    output_paths: Dict[str, Path],
    device: str,
    classifier: Optional[TeamClassifier],
    rosters: Optional[Dict[int, Dict[str, Dict[str, Optional[str]]]]] = None,
) -> None:
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    frame_generator = sv.get_video_frames_generator(str(video_path))

    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=15)
    transformer: Optional[ViewTransformer] = None
    last_ball_pitch: Optional[np.ndarray] = None
    last_ball_frame: Optional[int] = None
    last_ball_owner: Optional[dict] = None
    last_ball_speed: float = 0.0
    ball_gaps: List[Dict[str, int]] = []
    active_gap: Optional[Dict[str, int]] = None
    jersey_memory: Dict[int, str] = {}

    with output_paths["telemetry"].open("w") as fp, tqdm(
        frame_generator, total=video_info.total_frames, desc="logging", unit="frame"
    ) as iterator:
        for frame_idx, frame in enumerate(iterator):
            timestamp = frame_idx / video_info.fps if video_info.fps else 0.0
            timestamp_str = f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"

            # Pitch keypoints → View transformer
            pitch_res = models["pitch"](frame, verbose=False, device=device)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_res)
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            if mask.sum() >= 4:
                transformer = ViewTransformer(
                    source=keypoints.xy[0][mask].astype(np.float32),
                    target=np.array(CONFIG.vertices)[mask].astype(np.float32),
                )

            # Player detections + tracking
            player_res = models["player"](frame, imgsz=1280, verbose=False, device=device)[0]
            detections = sv.Detections.from_ultralytics(player_res)
            detections = tracker.update_with_detections(detections)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            player_crops = get_crops(frame, players) if len(players) > 0 else []
            update_jersey_memory(players, player_crops, jersey_memory)
            if classifier and len(player_crops) > 0:
                team_ids = classifier.predict(player_crops)
            elif len(player_crops) > 0:
                anchors = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                frame_width = frame.shape[1]
                team_ids = (anchors[:, 0] > frame_width / 2).astype(int)
            else:
                team_ids = np.array([], dtype=int)

            goalkeeper_crops = get_crops(frame, goalkeepers) if len(goalkeepers) > 0 else []
            update_jersey_memory(goalkeepers, goalkeeper_crops, jersey_memory)

            if len(goalkeepers) > 0 and len(players) > 0:
                keeper_team = resolve_goalkeepers_team_id(players, team_ids, goalkeepers)
            else:
                keeper_team = np.zeros(len(goalkeepers), dtype=int)

            # Ball detection/smoothing (optimized - no slicer, direct detection)
            ball_res = models["ball"](frame, imgsz=640, verbose=False, device=device)[0]
            ball_det = sv.Detections.from_ultralytics(ball_res)
            ball_det = ball_det[ball_det.class_id == BALL_CLASS_ID]
            ball_det = ball_tracker.update(ball_det)

            ball_entry: Dict[str, Any] = {
                "visible": len(ball_det) > 0,
                "image": None,
                "pitch": None,
                "confidence": None,
                "velocity": None,
                "speed": None,
                "owner_id": None,
                "owner_team": None,
            }
            if len(ball_det) > 0:
                anchor = ball_det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
                ball_entry["image"] = anchor.tolist()
                if ball_det.confidence is not None:
                    ball_entry["confidence"] = float(ball_det.confidence[0])
                if transformer is not None:
                    pitch_xy = transformer.transform_points(
                        np.asarray([anchor], dtype=np.float32)
                    )[0]
                    ball_entry["pitch"] = pitch_xy.tolist()
                    if last_ball_pitch is not None and last_ball_frame is not None:
                        dt = frame_idx - last_ball_frame
                        if dt > 0:
                            velocity = (pitch_xy - last_ball_pitch) / dt
                            speed = float(np.linalg.norm(velocity))
                            ball_entry["velocity"] = velocity.tolist()
                            ball_entry["speed"] = speed
                            last_ball_speed = speed
                    last_ball_pitch = pitch_xy
                    last_ball_frame = frame_idx

            players_serialized = serialize_entities(players, team_ids, transformer)
            goalkeepers_serialized = serialize_entities(goalkeepers, keeper_team, transformer)
            referees_serialized = serialize_entities(referees, np.full(len(referees), fill_value=2), transformer)

            enrich_entities_with_roster(players_serialized, jersey_memory, rosters)
            enrich_entities_with_roster(goalkeepers_serialized, jersey_memory, rosters)

            owner_candidate = None
            if ball_entry["pitch"] is not None:
                owner_candidate = nearest_owner(
                    np.array(ball_entry["pitch"], dtype=float),
                    players_serialized + goalkeepers_serialized,
                    radius_cm=450.0,
                )
            if owner_candidate:
                ball_entry["owner_id"] = owner_candidate["id"]
                ball_entry["owner_team"] = owner_candidate["team"]
                last_ball_owner = owner_candidate
            elif last_ball_owner:
                ball_entry["owner_team"] = last_ball_owner["team"]

            if ball_entry["visible"]:
            if active_gap is not None:
                active_gap["end_frame"] = frame_idx
                ball_gaps.append(active_gap)
                active_gap = None
            else:
                should_track_gap = last_ball_speed > 0.5 or last_ball_owner is not None
                if should_track_gap:
                    if active_gap is None:
                        active_gap = {"start_frame": frame_idx, "end_frame": frame_idx}
                    else:
                        active_gap["end_frame"] = frame_idx

            # Draw annotations and save frames
            annotated_frame = frame.copy()
            
            # Combine all entities for annotation
            all_entities = sv.Detections.empty()
            all_teams = np.array([], dtype=int)
            all_labels = []
            
            if len(players) > 0:
                all_entities = sv.Detections.merge([all_entities, players])
                all_teams = np.concatenate([all_teams, team_ids])
                for p in players_serialized:
                    all_labels.append(f"P{p['id']} T{p['team']}")
            
            if len(goalkeepers) > 0:
                all_entities = sv.Detections.merge([all_entities, goalkeepers])
                all_teams = np.concatenate([all_teams, keeper_team])
                for g in goalkeepers_serialized:
                    all_labels.append(f"GK{g['id']} T{g['team']}")
            
            if len(ball_det) > 0:
                all_entities = sv.Detections.merge([all_entities, ball_det])
            
            # Draw players and goalkeepers with ellipses
            if len(players) > 0 or len(goalkeepers) > 0:
                entity_detections = sv.Detections.merge([players, goalkeepers])
                entity_teams = np.concatenate([team_ids, keeper_team]) if len(team_ids) > 0 or len(keeper_team) > 0 else np.array([], dtype=int)
                if len(entity_teams) > 0:
                    color_lookup = np.array([entity_teams[i] if i < len(entity_teams) else 0 for i in range(len(entity_detections))])
                    annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, entity_detections, custom_color_lookup=color_lookup)
                    if len(all_labels) > 0:
                        labels = [all_labels[i] if i < len(all_labels) else "" for i in range(len(entity_detections))]
                        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, entity_detections, labels=labels)
            
            # Draw ball with triangle
            if len(ball_det) > 0:
                annotated_frame = TRIANGLE_ANNOTATOR.annotate(annotated_frame, ball_det)
            
            # Save annotated frame
            frame_filename = output_paths["frames_dir"] / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_filename), annotated_frame)
            
            # Also save if ball is missing
            if not ball_entry["visible"]:
                missing_frame_filename = output_paths["missing_ball_dir"] / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(missing_frame_filename), annotated_frame)

            frame_record = {
                "frame": frame_idx,
                "timestamp": timestamp_str,
                "ball": ball_entry,
                "players": players_serialized,
                "goalkeepers": goalkeepers_serialized,
                "referees": referees_serialized,
            }
            fp.write(json.dumps(frame_record) + "\n")

    if active_gap is not None:
        ball_gaps.append(active_gap)

    output_paths["gaps"].write_text(json.dumps(ball_gaps, indent=2))
    print(f"[telemetry] Ball gap windows -> {output_paths['gaps']}")
    print(f"[telemetry] Telemetry -> {output_paths['telemetry']}")


def main() -> None:
    args = parse_args()
    video_path = Path(args.source_video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"No such video: {video_path}")

    paths = ensure_output_paths(video_path, Path(args.output_dir) if args.output_dir else None)
    models = load_models(device=args.device)
    crops = collect_player_crops(models["player"], video_path, args.sample_stride)
    classifier = fit_team_classifier(args.device, crops)
    if classifier:
        print("[telemetry] Team classifier trained.")
    else:
        print("[telemetry] Falling back to position-based team heuristic.")

    roster_entries: Dict[str, List[RosterEntry]] = {}
    roster_lookups: Dict[int, Dict[str, Dict[str, Optional[str]]]] = {}
    if args.home_lineup_path:
        home_entries = load_lineup_csv(Path(args.home_lineup_path), args.home_team_name, team_id=0)
        roster_entries["home"] = home_entries
        roster_lookups[0] = roster_to_lookup(home_entries)
        print(f"[telemetry] Loaded {len(home_entries)} home lineup entries.")
    if args.away_lineup_path:
        away_entries = load_lineup_csv(Path(args.away_lineup_path), args.away_team_name, team_id=1)
        roster_entries["away"] = away_entries
        roster_lookups[1] = roster_to_lookup(away_entries)
        print(f"[telemetry] Loaded {len(away_entries)} away lineup entries.")

    metadata = {
        "video": str(video_path),
        "fps": sv.VideoInfo.from_video_path(str(video_path)).fps,
        "player_model": PLAYER_DETECTION_MODEL_PATH,
        "ball_model": BALL_DETECTION_MODEL_PATH,
        "pitch_model": PITCH_DETECTION_MODEL_PATH,
        "team_classifier_trained": classifier is not None,
        "stride": args.sample_stride,
        "gap_file": str(paths["gaps"]),
        "lineups": {
            team: [entry.to_dict() for entry in entries]
            for team, entries in roster_entries.items()
        },
    }
    write_metadata(paths["metadata"], metadata)
    log_frames(models, video_path, paths, args.device, classifier, roster_lookups)


if __name__ == "__main__":
    main()


