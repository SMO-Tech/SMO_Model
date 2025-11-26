"""
Refine ball tracks by re-running detection over gap windows.

Usage:
    python tools/refine_ball_tracks.py \
        --video_path data/0bfacc_0.mp4 \
        --telemetry_path analysis/0bfacc_0/telemetry.jsonl \
        --metadata_path analysis/0bfacc_0/metadata.json \
        --gaps_path analysis/0bfacc_0/ball_gap_windows.json \
        --output_path analysis/0bfacc_0/telemetry_refined.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from sports.common.view import ViewTransformer
from main import (  # type: ignore
    BALL_CLASS_ID,
    BALL_DETECTION_MODEL_PATH,
    CONFIG,
    PITCH_DETECTION_MODEL_PATH,
    TRIANGLE_ANNOTATOR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine ball telemetry on gap windows.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--telemetry_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--gaps_path", type=str, required=True)
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output telemetry path (defaults to *_refined.jsonl).",
    )
    parser.add_argument("--padding", type=int, default=3, help="Frames to extend around each gap.")
    parser.add_argument("--ball_conf", type=float, default=0.15, help="Confidence threshold for rerun.")
    parser.add_argument(
        "--save_recovered_frames",
        action="store_true",
        help="Save frames where ball was recovered to analysis/<video>/frames_recovered/",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    entries: List[dict] = []
    with path.open() as fp:
        for line in fp:
            entries.append(json.loads(line))
    return entries


def write_jsonl(path: Path, entries: List[dict]) -> None:
    with path.open("w") as fp:
        for entry in entries:
            fp.write(json.dumps(entry) + "\n")


def main() -> None:
    args = parse_args()
    video_path = Path(args.video_path)
    telemetry_path = Path(args.telemetry_path)
    metadata_path = Path(args.metadata_path)
    gaps_path = Path(args.gaps_path)
    if not (video_path.exists() and telemetry_path.exists() and metadata_path.exists() and gaps_path.exists()):
        raise FileNotFoundError("One or more required files are missing.")

    telemetry = load_jsonl(telemetry_path)
    metadata = json.loads(metadata_path.read_text())
    gaps: List[Dict[str, int]] = json.loads(gaps_path.read_text())

    output_path = Path(args.output_path) if args.output_path else telemetry_path.with_name("telemetry_refined.jsonl")
    output_dir = output_path.parent

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    total_frames = video_info.total_frames

    ball_model = YOLO(BALL_DETECTION_MODEL_PATH)
    pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")

    recovered_frames_dir = None
    if args.save_recovered_frames:
        recovered_frames_dir = output_dir / "frames_recovered"
        recovered_frames_dir.mkdir(exist_ok=True)

    recovered_frames = 0

    for gap in gaps:
        start = max(0, gap.get("start_frame", 0) - args.padding)
        end = min(total_frames - 1, gap.get("end_frame", 0) + args.padding)
        for frame_idx in range(start, end + 1):
            if frame_idx >= len(telemetry):
                break
            ball_data = telemetry[frame_idx]["ball"]
            if ball_data.get("pitch") is not None:
                continue  # already present

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            # Build transformer for this frame
            pitch_res = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_res)
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            if mask.sum() < 4:
                continue
            transformer = ViewTransformer(
                source=keypoints.xy[0][mask].astype(np.float32),
                target=np.array(CONFIG.vertices)[mask].astype(np.float32),
            )

            ball_res = ball_model(frame, imgsz=640, conf=args.ball_conf, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(ball_res)
            detections = detections[detections.class_id == BALL_CLASS_ID]
            if len(detections) == 0:
                continue

            anchor = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
            pitch_xy = transformer.transform_points(np.asarray([anchor], dtype=np.float32))[0]
            telemetry[frame_idx]["ball"]["image"] = anchor.tolist()
            telemetry[frame_idx]["ball"]["pitch"] = pitch_xy.tolist()
            telemetry[frame_idx]["ball"]["confidence"] = float(detections.confidence[0]) if detections.confidence is not None else None
            telemetry[frame_idx]["ball"]["recovered"] = True
            recovered_frames += 1

            # Save recovered frame with ball annotation
            if recovered_frames_dir is not None:
                annotated_frame = frame.copy()
                annotated_frame = TRIANGLE_ANNOTATOR.annotate(annotated_frame, detections)
                recovered_frame_path = recovered_frames_dir / f"frame_{frame_idx:06d}_recovered.jpg"
                cv2.imwrite(str(recovered_frame_path), annotated_frame)

            # Save recovered frame with ball annotation
            if recovered_frames_dir is not None:
                annotated_frame = frame.copy()
                annotated_frame = TRIANGLE_ANNOTATOR.annotate(annotated_frame, detections)
                recovered_frame_path = recovered_frames_dir / f"frame_{frame_idx:06d}_recovered.jpg"
                cv2.imwrite(str(recovered_frame_path), annotated_frame)

    cap.release()
    write_jsonl(output_path, telemetry)
    print(f"[refine] Recovered ball coordinates on {recovered_frames} frames -> {output_path}")


if __name__ == "__main__":
    main()


