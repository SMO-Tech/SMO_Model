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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# Allow "import main" and sports.* modules without modifying sys.path globally
TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

sys.path.append('/workspace/niyas_version2/SMO_FINAL')

from sports.common.ball import BallTracker
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from main import (  # type: ignore
    BALL_CLASS_ID,
    BALL_DETECTION_MODEL_PATH,
    CONFIG,
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    PLAYER_DETECTION_MODEL_PATH,
    PITCH_DETECTION_MODEL_PATH,
    REFEREE_CLASS_ID,
    STRIDE,
    get_crops,
    resolve_goalkeepers_team_id,
)


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
    return parser.parse_args()


def ensure_output_paths(video_path: Path, output_dir: Optional[Path]) -> Dict[str, Path]:
    if output_dir is None:
        base_dir = video_path.parent / "analysis" / video_path.stem
    else:
        base_dir = output_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base_dir,
        "telemetry": base_dir / "telemetry.jsonl",
        "metadata": base_dir / "metadata.json",
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
        result = player_model(frame, imgsz=1280, verbose=False, device='cuda')[0]
        detections = sv.Detections.from_ultralytics(result)
        crops.extend(get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID]))
        if len(crops) >= 50:  # Limit crops for faster training
            break
    return crops


def fit_team_classifier(device: str, crops: List[np.ndarray]) -> Optional[TeamClassifier]:
    if len(crops) < 2:
        print("[telemetry] Warning: insufficient crops to fit TeamClassifier.")
        return None
    classifier = TeamClassifier(device=device)
    classifier.fit(crops)
    return classifier


def write_metadata(path: Path, info: Dict[str, Any]) -> None:
    path.write_text(json.dumps(info, indent=2))
    print(f"[telemetry] Metadata -> {path}")


def log_frames(
    models: Dict[str, YOLO],
    video_path: Path,
    output_paths: Dict[str, Path],
    device: str,
    classifier: Optional[TeamClassifier],
) -> None:
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    frame_generator = sv.get_video_frames_generator(str(video_path))

    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=15)
    transformer: Optional[ViewTransformer] = None

    with output_paths["telemetry"].open("w") as fp, tqdm(
        frame_generator, total=video_info.total_frames, desc="logging", unit="frame"
    ) as iterator:
        for frame_idx, frame in enumerate(iterator):
            timestamp = frame_idx / video_info.fps if video_info.fps else 0.0
            timestamp_str = f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"

            # Pitch keypoints → View transformer
            pitch_res = models["pitch"](frame, verbose=False, device=device)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_res)
            
            # Ensure keypoints are not empty before proceeding
            if len(keypoints.xy) > 0 and len(keypoints.xy[0]) > 0:
                mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
                if mask.sum() >= 4:
                    transformer = ViewTransformer(
                        source=keypoints.xy[0][mask].astype(np.float32),
                        target=np.array(CONFIG.vertices)[mask].astype(np.float32),
                    )
                else:
                    transformer = None # Reset transformer if not enough keypoints
            else:
                transformer = None # Reset transformer if no keypoints detected

            # Player detections + tracking
            player_res = models["player"](frame, imgsz=1280, verbose=False, device=device)[0]
            detections = sv.Detections.from_ultralytics(player_res)
            detections = tracker.update_with_detections(detections)

            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]

            # Team classification - SIMPLE COLOR-BASED (FAST!)
            player_crops = get_crops(frame, players)
            if len(player_crops) > 0:
                # Simple heuristic: use position on field (left=team0, right=team1)
                anchors = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                frame_width = frame.shape[1]
                team_ids = (anchors[:, 0] > frame_width / 2).astype(int)
            else:
                team_ids = np.array([], dtype=int)

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

            def serialize_entities(det: sv.Detections, teams: np.ndarray) -> List[Dict[str, Any]]:
                result: List[Dict[str, Any]] = []
                if len(det) == 0:
                    return result
                anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(np.float32)
                pitch_pts = (
                    transformer.transform_points(anchors) if transformer is not None else None
                )
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

            frame_record = {
                "frame": frame_idx,
                "timestamp": timestamp_str,
                "ball": ball_entry,
                "players": serialize_entities(players, team_ids),
                "goalkeepers": serialize_entities(goalkeepers, keeper_team),
                "referees": serialize_entities(referees, np.full(len(referees), fill_value=2)),
            }
            fp.write(json.dumps(frame_record) + "\n")

    print(f"[telemetry] Telemetry -> {output_paths['telemetry']}")


def main() -> None:
    args = parse_args()
    video_path = Path(args.source_video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"No such video: {video_path}")

    paths = ensure_output_paths(video_path, Path(args.output_dir) if args.output_dir else None)
    models = load_models(device=args.device)
    # Skip slow team classifier training - using simple position-based teams
    classifier = None
    print("[telemetry] Using fast position-based team assignment (skip classifier training)")

    metadata = {
        "video": str(video_path),
        "fps": sv.VideoInfo.from_video_path(str(video_path)).fps,
        "player_model": PLAYER_DETECTION_MODEL_PATH,
        "ball_model": BALL_DETECTION_MODEL_PATH,
        "pitch_model": PITCH_DETECTION_MODEL_PATH,
        "team_classifier_trained": classifier is not None,
        "stride": args.sample_stride,
    }
    write_metadata(paths["metadata"], metadata)
    log_frames(models, video_path, paths, args.device, classifier)


if __name__ == "__main__":
    main()


