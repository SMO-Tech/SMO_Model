"""
Fast Telemetry Logger - GPU Optimized Version

This version uses:
- FP16 (half precision) for faster inference
- Batch processing where possible
- Skip slow team classifier (use position-based heuristic)
- Skip frame saving (optional)
- Optimized YOLO settings

Usage:
    python tools/fast_telemetry_logger.py --source_video_path data/video.mp4 --device cuda
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
import torch

# Allow imports from parent
TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from sports.common.ball import BallTracker
from sports.common.view import ViewTransformer
from main import (
    BALL_CLASS_ID,
    BALL_DETECTION_MODEL_PATH,
    CONFIG,
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    PLAYER_DETECTION_MODEL_PATH,
    PITCH_DETECTION_MODEL_PATH,
    REFEREE_CLASS_ID,
    resolve_goalkeepers_team_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast GPU-optimized telemetry logger.")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
    parser.add_argument("--half", action="store_true", default=True, help="Use FP16 half precision")
    parser.add_argument("--save_frames", action="store_true", default=False, help="Save annotated frames (slower)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    return parser.parse_args()


def ensure_output_paths(video_path: Path, output_dir: Optional[Path]) -> Dict[str, Path]:
    if output_dir is None:
        base_dir = video_path.parent / "analysis" / video_path.stem
    else:
        base_dir = Path(output_dir) / video_path.stem
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base_dir,
        "telemetry": base_dir / "telemetry.jsonl",
        "metadata": base_dir / "metadata.json",
        "gaps": base_dir / "ball_gap_windows.json",
    }


def load_models_optimized(device: str, use_half: bool = True) -> Dict[str, YOLO]:
    """Load models with GPU optimization"""
    print(f"[fast-telemetry] Loading models on {device} (half={use_half})...")
    
    models = {}
    for name, path in [
        ("player", PLAYER_DETECTION_MODEL_PATH),
        ("pitch", PITCH_DETECTION_MODEL_PATH),
        ("ball", BALL_DETECTION_MODEL_PATH),
    ]:
        model = YOLO(path)
        model.to(device)
        
        # Note: YOLO handles half precision internally during inference
        # We pass half=True to the predict call instead of converting model
        
        models[name] = model
        print(f"  ✓ {name} model loaded")
    
    return models


def nearest_owner(
    ball_pitch: np.ndarray,
    players: List[dict],
    radius_cm: float = 450.0,
) -> Optional[dict]:
    """Find nearest player to ball"""
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
            candidate = {"id": pid, "team": team, "dist": dist}
            best_dist = dist
    return candidate


def serialize_entities(
    det: sv.Detections,
    teams: np.ndarray,
    transformer: Optional[ViewTransformer],
) -> List[Dict[str, Any]]:
    """Serialize detections to dict format"""
    result = []
    if len(det) == 0:
        return result
    
    anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(np.float32)
    pitch_pts = transformer.transform_points(anchors) if transformer is not None else None
    
    for idx, tid in enumerate(det.tracker_id):
        entry = {
            "id": int(tid) if tid is not None else None,
            "team": int(teams[idx]) if idx < len(teams) else 0,
            "confidence": float(det.confidence[idx]) if det.confidence is not None else None,
            "bbox": det.xyxy[idx].astype(float).tolist(),
            "image": anchors[idx].astype(float).tolist(),
            "pitch": pitch_pts[idx].astype(float).tolist() if pitch_pts is not None else None,
        }
        result.append(entry)
    return result


def process_video_fast(
    models: Dict[str, YOLO],
    video_path: Path,
    output_paths: Dict[str, Path],
    device: str,
    use_half: bool = True,
) -> None:
    """Fast video processing with GPU optimization"""
    
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    frame_generator = sv.get_video_frames_generator(str(video_path))
    
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)
    transformer: Optional[ViewTransformer] = None
    
    last_ball_pitch: Optional[np.ndarray] = None
    last_ball_frame: Optional[int] = None
    last_ball_owner: Optional[dict] = None
    last_ball_speed: float = 0.0
    ball_gaps: List[Dict[str, int]] = []
    active_gap: Optional[Dict[str, int]] = None
    
    with output_paths["telemetry"].open("w") as fp:
        pbar = tqdm(
            enumerate(frame_generator),
            total=video_info.total_frames,
            desc="Processing",
            unit="frame"
        )
        
        for frame_idx, frame in pbar:
            timestamp = frame_idx / video_info.fps if video_info.fps else 0.0
            timestamp_str = f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
            
            # Pitch detection (for view transformer)
            pitch_res = models["pitch"](
                frame, 
                verbose=False, 
                device=device,
                half=use_half and device == "cuda"
            )[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_res)
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            if mask.sum() >= 4:
                transformer = ViewTransformer(
                    source=keypoints.xy[0][mask].astype(np.float32),
                    target=np.array(CONFIG.vertices)[mask].astype(np.float32),
                )
            
            # Player detection with optimized settings
            player_res = models["player"](
                frame,
                imgsz=1280,
                verbose=False,
                device=device,
                half=use_half and device == "cuda"
            )[0]
            detections = sv.Detections.from_ultralytics(player_res)
            detections = tracker.update_with_detections(detections)
            
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]
            
            # Fast team assignment (position-based, no slow embedding extraction)
            if len(players) > 0:
                anchors = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                frame_width = frame.shape[1]
                team_ids = (anchors[:, 0] > frame_width / 2).astype(int)
            else:
                team_ids = np.array([], dtype=int)
            
            if len(goalkeepers) > 0 and len(players) > 0:
                keeper_team = resolve_goalkeepers_team_id(players, team_ids, goalkeepers)
            else:
                keeper_team = np.zeros(len(goalkeepers), dtype=int)
            
            # Ball detection with optimized settings
            ball_res = models["ball"](
                frame,
                imgsz=640,
                conf=0.10,
                verbose=False,
                device=device,
                half=use_half and device == "cuda"
            )[0]
            ball_det = sv.Detections.from_ultralytics(ball_res)
            ball_det = ball_det[ball_det.class_id == BALL_CLASS_ID]
            ball_det = ball_tracker.update(ball_det)
            
            # Build ball entry
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
            
            # Serialize entities
            players_serialized = serialize_entities(players, team_ids, transformer)
            goalkeepers_serialized = serialize_entities(goalkeepers, keeper_team, transformer)
            referees_serialized = serialize_entities(
                referees, np.full(len(referees), fill_value=2), transformer
            )
            
            # Find ball owner
            if ball_entry["pitch"] is not None:
                owner = nearest_owner(
                    np.array(ball_entry["pitch"], dtype=float),
                    players_serialized + goalkeepers_serialized,
                )
                if owner:
                    ball_entry["owner_id"] = owner["id"]
                    ball_entry["owner_team"] = owner["team"]
                    last_ball_owner = owner
                elif last_ball_owner:
                    ball_entry["owner_team"] = last_ball_owner["team"]
            
            # Track ball gaps
            if ball_entry["visible"]:
                if active_gap is not None:
                    active_gap["end_frame"] = frame_idx
                    ball_gaps.append(active_gap)
                    active_gap = None
            else:
                if last_ball_speed > 0.5 or last_ball_owner is not None:
                    if active_gap is None:
                        active_gap = {"start_frame": frame_idx, "end_frame": frame_idx}
                    else:
                        active_gap["end_frame"] = frame_idx
            
            # Write frame record
            frame_record = {
                "frame": frame_idx,
                "timestamp": timestamp_str,
                "ball": ball_entry,
                "players": players_serialized,
                "goalkeepers": goalkeepers_serialized,
                "referees": referees_serialized,
            }
            fp.write(json.dumps(frame_record) + "\n")
            
            # Update progress bar with FPS
            if frame_idx > 0:
                fps = pbar.format_dict.get('rate', 0) or 0
                pbar.set_postfix({"fps": f"{fps:.1f}"})
    
    # Save remaining gap
    if active_gap is not None:
        ball_gaps.append(active_gap)
    
    output_paths["gaps"].write_text(json.dumps(ball_gaps, indent=2))
    print(f"[fast-telemetry] ✅ Telemetry -> {output_paths['telemetry']}")
    print(f"[fast-telemetry] ✅ Ball gaps ({len(ball_gaps)}) -> {output_paths['gaps']}")


def interpolate_ball_positions(telemetry_path: Path, max_gap: int = 10) -> int:
    """Interpolate missing ball positions for short gaps"""
    with open(telemetry_path, 'r') as f:
        records = [json.loads(line) for line in f]
    
    interpolated_count = 0
    i = 0
    while i < len(records):
        if records[i]["ball"].get("pitch"):
            start_idx = i
            start_pos = np.array(records[i]["ball"]["pitch"])
            
            j = i + 1
            while j < len(records) and not records[j]["ball"].get("pitch"):
                j += 1
            
            if j < len(records) and (j - i) <= max_gap:
                end_pos = np.array(records[j]["ball"]["pitch"])
                gap_size = j - i
                
                for k in range(1, gap_size):
                    alpha = k / gap_size
                    interp_pos = start_pos + alpha * (end_pos - start_pos)
                    records[i + k]["ball"]["pitch"] = interp_pos.tolist()
                    records[i + k]["ball"]["visible"] = True
                    records[i + k]["ball"]["interpolated"] = True
                    interpolated_count += 1
                
                i = j
            else:
                i += 1
        else:
            i += 1
    
    with open(telemetry_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    return interpolated_count


def main() -> None:
    args = parse_args()
    video_path = Path(args.source_video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[fast-telemetry] ⚠️ CUDA not available, falling back to CPU")
        args.device = "cpu"
        args.half = False
    
    if args.device == "cuda":
        print(f"[fast-telemetry] 🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[fast-telemetry] 📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    paths = ensure_output_paths(video_path, Path(args.output_dir) if args.output_dir else None)
    models = load_models_optimized(args.device, args.half)
    
    # Write metadata
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    metadata = {
        "video": str(video_path),
        "fps": video_info.fps,
        "total_frames": video_info.total_frames,
        "resolution": f"{video_info.width}x{video_info.height}",
        "player_model": PLAYER_DETECTION_MODEL_PATH,
        "ball_model": BALL_DETECTION_MODEL_PATH,
        "pitch_model": PITCH_DETECTION_MODEL_PATH,
        "device": args.device,
        "half_precision": args.half,
        "gap_file": str(paths["gaps"]),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2))
    
    # Process video
    process_video_fast(models, video_path, paths, args.device, args.half)
    
    # Interpolate short gaps
    interpolated = interpolate_ball_positions(paths["telemetry"], max_gap=10)
    print(f"[fast-telemetry] ✅ Interpolated {interpolated} frames")


if __name__ == "__main__":
    main()

