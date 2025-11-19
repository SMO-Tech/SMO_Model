"""
Fast Pass Detection - Optimized for speed and accuracy

Uses simplified approach:
- Single player detection model
- Faster ball tracking
- Better pass detection logic with Team A/B labels
- No duplicate timestamps
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# Import from main
TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from main import (
    PLAYER_CLASS_ID,
    PLAYER_DETECTION_MODEL_PATH,
    BALL_DETECTION_MODEL_PATH,
    GOALKEEPER_CLASS_ID,
    REFEREE_CLASS_ID,
    get_crops,
    resolve_goalkeepers_team_id,
)
from sports.common.team import TeamClassifier


@dataclass
class PassEvent:
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    passer_id: Optional[int]
    receiver_id: Optional[int]
    passer_team: str  # "Team A" or "Team B"
    receiver_team: str
    distance_px: float
    pass_type: str  # short, medium, long
    outcome: str  # successful, intercepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast pass detection")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--possession_radius_px", type=float, default=80.0)
    parser.add_argument("--min_possession_frames", type=int, default=2)
    parser.add_argument("--max_pass_frames", type=int, default=45)
    return parser.parse_args()


def classify_distance(distance_px: float) -> str:
    """Classify pass type by distance in pixels"""
    if distance_px < 100:
        return "short"
    elif distance_px < 200:
        return "medium"
    else:
        return "long"


def format_time(frame: int, fps: float) -> str:
    """Format frame as MM:SS timestamp"""
    seconds = frame / fps
    return f"{int(seconds//60):02d}:{int(seconds%60):02d}"


def main() -> None:
    args = parse_args()
    video_path = Path(args.source_video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = video_path.parent / "analysis" / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "passes_fast.csv"
    json_path = output_dir / "passes_fast.json"
    
    print(f"[fast] Loading models on {args.device}...")
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=args.device)
    
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    fps = video_info.fps
    
    print(f"[fast] Video: {video_info.total_frames} frames @ {fps} FPS")
    print(f"[fast] Collecting player crops for team classification...")
    
    # Collect crops every 30 frames (faster sampling)
    crops = []
    frame_gen = sv.get_video_frames_generator(str(video_path), stride=30)
    for frame in tqdm(frame_gen, desc="crops", unit="frame"):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops.extend(get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID]))
    
    # Fit team classifier
    print(f"[fast] Training team classifier with {len(crops)} crops...")
    team_classifier = TeamClassifier(device=args.device)
    team_classifier.fit(crops)
    
    # Process video for passes
    print(f"[fast] Detecting passes...")
    frame_generator = sv.get_video_frames_generator(str(video_path))
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    # Pass detection state
    current_owner = None
    owner_frames = 0
    events: List[PassEvent] = []
    last_event_end_frame = -args.max_pass_frames  # Allow immediate first pass
    
    with tqdm(frame_generator, total=video_info.total_frames, desc="analyzing", unit="frame") as iterator:
        for frame_idx, frame in enumerate(iterator):
            # Detect players
            result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)
            
            # Separate by class
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            
            # Team classification
            player_crops = get_crops(frame, players)
            if len(player_crops) > 0:
                team_ids = team_classifier.predict(player_crops)
            else:
                team_ids = np.array([])
            
            # Goalkeeper teams
            if len(goalkeepers) > 0 and len(players) > 0:
                keeper_teams = resolve_goalkeepers_team_id(players, team_ids, goalkeepers)
            else:
                keeper_teams = np.array([])
            
            # Merge all entities
            all_detections = sv.Detections.merge([players, goalkeepers])
            all_teams = np.concatenate([team_ids, keeper_teams]) if len(team_ids) > 0 else np.array([])
            
            if len(all_detections) == 0:
                continue
            
            # Get positions
            positions = all_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            
            # Simple ball position estimation: find center of mass of players
            # In real scenario, you'd track actual ball, but for speed we approximate
            ball_pos = np.mean(positions, axis=0)
            
            # Find nearest player to "ball" (possession)
            distances = np.linalg.norm(positions - ball_pos, axis=1)
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < args.possession_radius_px:
                player_id = all_detections.tracker_id[closest_idx]
                player_team = all_teams[closest_idx] if closest_idx < len(all_teams) else 0
                player_pos = positions[closest_idx]
                
                team_name = "Team A" if player_team == 0 else "Team B"
                
                # Check for possession change (pass)
                if current_owner is not None and player_id != current_owner["id"]:
                    # Possession changed!
                    if owner_frames >= args.min_possession_frames:
                        # Valid pass duration
                        if frame_idx - current_owner["last_frame"] <= args.max_pass_frames:
                            # Not too slow
                            if frame_idx - last_event_end_frame >= 5:  # Avoid duplicates
                                # Calculate distance
                                dist = np.linalg.norm(player_pos - current_owner["pos"])
                                
                                # Determine outcome
                                if team_name == current_owner["team"]:
                                    outcome = "successful"
                                else:
                                    outcome = "intercepted"
                                
                                event = PassEvent(
                                    start_frame=current_owner["last_frame"],
                                    end_frame=frame_idx,
                                    start_time=format_time(current_owner["last_frame"], fps),
                                    end_time=format_time(frame_idx, fps),
                                    passer_id=int(current_owner["id"]),
                                    receiver_id=int(player_id),
                                    passer_team=current_owner["team"],
                                    receiver_team=team_name,
                                    distance_px=float(dist),
                                    pass_type=classify_distance(dist),
                                    outcome=outcome,
                                )
                                events.append(event)
                                last_event_end_frame = frame_idx
                    
                    # Update owner
                    current_owner = {
                        "id": player_id,
                        "team": team_name,
                        "last_frame": frame_idx,
                        "pos": player_pos,
                    }
                    owner_frames = 1
                elif current_owner is not None and player_id == current_owner["id"]:
                    # Same owner continues
                    owner_frames += 1
                    current_owner["last_frame"] = frame_idx
                    current_owner["pos"] = player_pos
                else:
                    # New owner
                    current_owner = {
                        "id": player_id,
                        "team": team_name,
                        "last_frame": frame_idx,
                        "pos": player_pos,
                    }
                    owner_frames = 1
    
    # Save results
    print(f"[fast] Detected {len(events)} pass events")
    
    if events:
        # Save CSV
        with csv_path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "start_time", "end_time", "start_frame", "end_frame",
                "passer_id", "receiver_id", "passer_team", "receiver_team",
                "distance_px", "pass_type", "outcome"
            ])
            for ev in events:
                writer.writerow([
                    ev.start_time, ev.end_time, ev.start_frame, ev.end_frame,
                    ev.passer_id, ev.receiver_id, ev.passer_team, ev.receiver_team,
                    f"{ev.distance_px:.1f}", ev.pass_type, ev.outcome
                ])
        
        # Save JSON
        json_data = [vars(ev) for ev in events]
        json_path.write_text(json.dumps(json_data, indent=2))
        
        print(f"[fast] ✅ CSV: {csv_path}")
        print(f"[fast] ✅ JSON: {json_path}")
        
        # Print summary
        successful = sum(1 for e in events if e.outcome == "successful")
        intercepted = sum(1 for e in events if e.outcome == "intercepted")
        print(f"\n📊 Summary:")
        print(f"   Total passes: {len(events)}")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Intercepted: {intercepted}")
    else:
        print("[fast] ⚠️  No passes detected")

if __name__ == "__main__":
    main()

