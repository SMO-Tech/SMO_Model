import argparse
import csv
import json
import os
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
import supervision as sv  # type: ignore[import-untyped]
import torch  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]
from ultralytics import YOLO  # type: ignore[import-untyped]

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


CLASS_NAME_LOOKUP = {
    BALL_CLASS_ID: "ball",
    GOALKEEPER_CLASS_ID: "goalkeeper",
    PLAYER_CLASS_ID: "player",
    REFEREE_CLASS_ID: "referee",
}

PASS_DISTANCE_THRESHOLD_PX = 250

# Detection settings
DEFAULT_BALL_IMGSZ = 960  # Higher resolution for ball detection
DEFAULT_PLAYER_IMGSZ = 1536  # Higher resolution for player detection
DEFAULT_BALL_CONF = 0.10  # Lower confidence threshold for ball
DEFAULT_PLAYER_CONF = 0.25  # Confidence threshold for players
DEFAULT_NMS_THRESHOLD = 0.1  # NMS threshold
SLICE_OVERLAP_RATIO = 0.25  # 25% overlap between slices


def setup_gpu_optimizations(device: str) -> None:
    """
    Configure GPU optimizations for faster inference.
    
    Args:
        device: Device string ('cpu', 'cuda', 'mps', etc.)
    """
    if 'cuda' in device or device.startswith('cuda'):
        # Enable TensorFloat-32 for Ampere GPUs (30xx, A100, etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True


def load_model(
    model_path: str,
    device: str,
) -> YOLO:
    """
    Load a YOLO model.
    
    Args:
        model_path: Path to the model weights.
        device: Device to load model on.
        
    Returns:
        YOLO: Loaded model.
    """
    model = YOLO(model_path)
    model.to(device=device)
    return model


def is_cuda_device(device: str) -> bool:
    """Check if device is CUDA."""
    return 'cuda' in device or device.startswith('cuda')


def create_empty_detections() -> sv.Detections:
    if hasattr(sv.Detections, "empty"):
        return sv.Detections.empty()  # type: ignore[attr-defined]
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int64),
    )


def ensure_output_dirs(base_dir: str) -> Dict[str, str]:
    tracked_dir = os.path.join(base_dir, 'tracked')
    missed_dir = os.path.join(base_dir, 'missed')
    os.makedirs(tracked_dir, exist_ok=True)
    os.makedirs(missed_dir, exist_ok=True)
    return {"tracked": tracked_dir, "missed": missed_dir}


def build_player_labels(detections: sv.Detections) -> List[str]:
    labels = []
    if len(detections) == 0:
        return labels
    tracker_ids = detections.tracker_id
    class_ids = detections.class_id
    for idx in range(len(detections)):
        class_id = int(class_ids[idx]) if class_ids is not None else -1
        tracker_id = tracker_ids[idx] if tracker_ids is not None else None
        class_name = CLASS_NAME_LOOKUP.get(class_id, "object")
        labels.append(
            f"{class_name} #{int(tracker_id)}" if tracker_id is not None else class_name
        )
    return labels


def serialize_player_tracks(detections: sv.Detections) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    if len(detections) == 0:
        return serialized
    bottom_centers = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    for idx, bbox in enumerate(detections.xyxy):
        tracker_id = None
        if detections.tracker_id is not None:
            tracker_idx = detections.tracker_id[idx]
            tracker_id = int(tracker_idx) if tracker_idx is not None else None
        class_id = None
        if detections.class_id is not None:
            class_id = int(detections.class_id[idx])
        if class_id not in {PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID}:
            continue
        serialized.append({
            "tracker_id": tracker_id,
            "class_id": class_id,
            "bbox": bbox.tolist(),
            "bottom_center": bottom_centers[idx].tolist(),
        })
    return serialized


def find_nearest_player(
    ball_center: Optional[List[float]],
    player_tracks: List[Dict[str, Any]],
    max_distance: float = 75.0
) -> Optional[int]:
    if ball_center is None or player_tracks == []:
        return None
    ball_xy = np.array(ball_center)
    min_distance = float('inf')
    best_tracker = None
    for track in player_tracks:
        player_xy = np.array(track["bottom_center"])
        distance = np.linalg.norm(ball_xy - player_xy)
        if distance < min_distance:
            min_distance = distance
            best_tracker = track["tracker_id"]
    if min_distance <= max_distance:
        return best_tracker
    return None


def detect_passes(
    frames_metadata: List[Dict[str, Any]],
    fps: float,
    distance_threshold_px: float = PASS_DISTANCE_THRESHOLD_PX
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    current_owner: Optional[int] = None
    potential_pass: Optional[Dict[str, Any]] = None
    last_logged_pass: Optional[Tuple[int, int]] = None  # (passer_id, receiver_id)

    for entry in frames_metadata:
        frame_index = entry["frame_index"]
        timestamp = entry["timestamp"]
        ball_center = entry.get("ball_center")
        player_tracks = entry.get("player_tracks", [])
        owner = find_nearest_player(ball_center, player_tracks)

        if owner is not None and current_owner is None:
            current_owner = owner
            continue

        if current_owner is None:
            current_owner = owner
            continue

        if owner == current_owner:
            potential_pass = None
            continue

        if owner is None and entry["status"] == "tracked":
            if potential_pass is None:
                potential_pass = {
                    "passer": current_owner,
                    "start_frame": frame_index,
                    "start_time": timestamp,
                    "start_pos": ball_center,
                }
            continue

        if owner is not None and owner != current_owner and potential_pass is not None:
            # Check if this is a duplicate of the last logged pass
            if last_logged_pass == (potential_pass["passer"], owner):
                # Same passer→receiver pair; skip duplicate
                continue
            
            distance = 0.0
            if potential_pass["start_pos"] and ball_center:
                distance = float(
                    np.linalg.norm(
                        np.array(potential_pass["start_pos"]) - np.array(ball_center)
                    )
                )
            duration = timestamp - potential_pass["start_time"]
            pass_type = "long" if distance >= distance_threshold_px else "short"
            events.append({
                "start_frame": potential_pass["start_frame"],
                "start_time": potential_pass["start_time"],
                "end_frame": frame_index,
                "end_time": timestamp,
                "passer_id": potential_pass["passer"],
                "receiver_id": owner,
                "distance_px": distance,
                "duration_s": duration,
                "pass_type": pass_type,
            })
            last_logged_pass = (potential_pass["passer"], owner)
            current_owner = owner
            potential_pass = None

    return events


def export_pass_events_to_csv(events: List[Dict[str, Any]], csv_path: str) -> None:
    if not events:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "start_frame", "start_time", "end_frame", "end_time",
                "passer_id", "receiver_id", "distance_px", "duration_s", "pass_type"
            ])
        return

    fieldnames = [
        "start_frame", "start_time", "end_frame", "end_time",
        "passer_id", "receiver_id", "distance_px", "duration_s", "pass_type"
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    BALL_DETECTION_RECOVERY = 'BALL_DETECTION_RECOVERY'
    COMBINED_DETECTION = 'COMBINED_DETECTION'
    FRAME_EXTRACTION = 'FRAME_EXTRACTION'  # Extract frames with annotations


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def create_inference_slicer(
    callback,
    slice_wh: Tuple[int, int] = (640, 640),
    overlap_ratio: float = SLICE_OVERLAP_RATIO
) -> sv.InferenceSlicer:
    """
    Create an InferenceSlicer with configurable overlap.
    
    Args:
        callback: Detection callback function.
        slice_wh: Slice width and height.
        overlap_ratio: Overlap ratio between slices (0-1).
        
    Returns:
        sv.InferenceSlicer: Configured slicer.
    """
    overlap_wh = (int(slice_wh[0] * overlap_ratio), int(slice_wh[1] * overlap_ratio))
    return sv.InferenceSlicer(
        callback=callback,
        slice_wh=slice_wh,
        overlap_wh=overlap_wh,
    )


def run_pitch_detection(
    source_video_path: str,
    device: str,
    high_accuracy: bool = False
) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        high_accuracy (bool): Use higher accuracy settings.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    setup_gpu_optimizations(device)
    pitch_detection_model = load_model(PITCH_DETECTION_MODEL_PATH, device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(
    source_video_path: str,
    device: str,
    high_accuracy: bool = False
) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        high_accuracy (bool): Use higher accuracy settings (1536px resolution).

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    setup_gpu_optimizations(device)
    player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    imgsz = DEFAULT_PLAYER_IMGSZ if high_accuracy else 1280
    conf = DEFAULT_PLAYER_CONF
    
    for frame in frame_generator:
        result = player_detection_model(
            frame, imgsz=imgsz, conf=conf, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        # Apply stricter NMS
        detections = detections.with_nms(threshold=DEFAULT_NMS_THRESHOLD)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(
    source_video_path: str,
    device: str,
    high_accuracy: bool = False
) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        high_accuracy (bool): Use higher accuracy settings (960px, ByteTrack).

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    setup_gpu_optimizations(device)
    ball_detection_model = load_model(BALL_DETECTION_MODEL_PATH, device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    # Enhanced ball tracker with velocity prediction
    ball_tracker = BallTracker(
        buffer_size=30,
        velocity_alpha=0.3,
        max_prediction_frames=5
    )
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    
    # ByteTrack for ball (consistent with player tracking)
    ball_bytetrack = sv.ByteTrack(
        minimum_consecutive_frames=3,
        lost_track_buffer=30,
    )
    
    imgsz = DEFAULT_BALL_IMGSZ if high_accuracy else 640
    conf = DEFAULT_BALL_CONF

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(
            image_slice, imgsz=imgsz, conf=conf, verbose=False
        )[0]
        return sv.Detections.from_ultralytics(result)

    # Use overlapping slices for better boundary detection
    slicer = create_inference_slicer(callback, slice_wh=(640, 640))

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=DEFAULT_NMS_THRESHOLD)
        
        # Apply ByteTrack for consistent tracking
        if high_accuracy:
            detections = ball_bytetrack.update_with_detections(detections)
        
        # Apply velocity-based tracker
        detections = ball_tracker.update(detections)
        
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection_with_recovery(
    source_video_path: str,
    target_video_path: str,
    device: str,
    output_dir: str,
    pass_csv_path: Optional[str] = None,
    recovery_imgsz: int = DEFAULT_BALL_IMGSZ,
    show_preview: bool = False,
    high_accuracy: bool = False
) -> None:
    """
    Run ball detection with a recovery pass and export annotated frames plus pass stats.
    """
    if not output_dir:
        raise ValueError("ball_output_dir is required for BALL_DETECTION_RECOVERY mode.")

    setup_gpu_optimizations(device)
    os.makedirs(output_dir, exist_ok=True)
    directories = ensure_output_dirs(output_dir)
    metadata_path = os.path.join(output_dir, "ball_detections.json")
    if pass_csv_path is None:
        pass_csv_path = os.path.join(output_dir, "pass_events.csv")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps if video_info.fps else 30.0

    ball_detection_model = load_model(BALL_DETECTION_MODEL_PATH, device)
    recovery_detection_model = load_model(BALL_DETECTION_MODEL_PATH, device)
    player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)

    base_imgsz = DEFAULT_BALL_IMGSZ if high_accuracy else 640
    player_imgsz = DEFAULT_PLAYER_IMGSZ if high_accuracy else 1280

    def base_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(
            image_slice, imgsz=base_imgsz, conf=DEFAULT_BALL_CONF, verbose=False
        )[0]
        return sv.Detections.from_ultralytics(result)

    slicer = create_inference_slicer(base_callback, slice_wh=(640, 640))

    first_pass_detections: List[sv.Detections] = []
    missed_indices: Set[int] = set()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    print(f"[Pass 1/2] Detecting ball in {video_info.total_frames} frames...")
    for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="Ball detection")):
        detections = slicer(frame).with_nms(threshold=DEFAULT_NMS_THRESHOLD)
        first_pass_detections.append(detections)
        if len(detections) == 0:
            missed_indices.add(frame_idx)
    
    print(f"[Pass 1/2] Ball detected in {len(first_pass_detections) - len(missed_indices)} frames, missed in {len(missed_indices)} frames.")

    # Enhanced ball tracker with velocity prediction
    ball_tracker = BallTracker(
        buffer_size=30,
        velocity_alpha=0.3,
        max_prediction_frames=5
    )
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    player_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    metadata: List[Dict[str, Any]] = []

    print(f"[Pass 2/2] Running recovery pass and saving annotated frames...")
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="Recovery & export")):
            ball_detections = (
                first_pass_detections[frame_idx]
                if frame_idx < len(first_pass_detections)
                else create_empty_detections()
            )
            if frame_idx in missed_indices:
                recovery_result = recovery_detection_model(
                    frame, imgsz=recovery_imgsz, conf=DEFAULT_BALL_CONF, verbose=False
                )[0]
                ball_detections = sv.Detections.from_ultralytics(recovery_result)

            tracked_ball = ball_tracker.update(ball_detections)
            player_result = player_detection_model(
                frame, imgsz=player_imgsz, conf=DEFAULT_PLAYER_CONF, verbose=False
            )[0]
            player_detections = sv.Detections.from_ultralytics(player_result)
            tracked_players = player_tracker.update_with_detections(player_detections)

            annotated_frame = frame.copy()
            annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, tracked_players)
            player_labels = build_player_labels(tracked_players)
            annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
                annotated_frame, tracked_players, labels=player_labels
            )
            status = 'tracked' if len(tracked_ball) > 0 else 'missed'
            if len(tracked_ball) > 0:
                annotated_frame = ball_annotator.annotate(annotated_frame, tracked_ball)
                ball_bbox = tracked_ball.xyxy.tolist()
                ball_centers = tracked_ball.get_anchors_coordinates(
                    sv.Position.CENTER
                ).tolist()
            else:
                ball_bbox = []
                ball_centers = []

            sink.write_frame(annotated_frame)
            if show_preview:
                cv2.imshow("frame", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            target_dir = directories['tracked'] if status == 'tracked' else directories['missed']
            cv2.imwrite(
                os.path.join(target_dir, f"frame_{frame_idx:06d}.jpg"),
                annotated_frame
            )

            metadata.append({
                "frame_index": frame_idx,
                "timestamp": frame_idx / fps,
                "status": status,
                "ball_bbox": ball_bbox,
                "ball_center": ball_centers[0] if ball_centers else None,
                "player_tracks": serialize_player_tracks(tracked_players),
            })

    if show_preview:
        cv2.destroyAllWindows()

    print(f"✓ Saved {len(metadata)} annotated frames.")
    print(f"✓ Writing metadata to {metadata_path}")
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"[Pass detection] Analyzing ball trajectories...")
    pass_events = detect_passes(metadata, fps)
    export_pass_events_to_csv(pass_events, pass_csv_path)
    print(f"✓ Exported {len(pass_events)} pass events to {pass_csv_path}")
    print(f"✓ Done! Check output in: {output_dir}")


def run_player_tracking(
    source_video_path: str,
    device: str,
    high_accuracy: bool = False
) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        high_accuracy (bool): Use higher accuracy settings.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    setup_gpu_optimizations(device)
    player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    imgsz = DEFAULT_PLAYER_IMGSZ if high_accuracy else 1280
    
    for frame in frame_generator:
        result = player_detection_model(
            frame, imgsz=imgsz, conf=DEFAULT_PLAYER_CONF, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=DEFAULT_NMS_THRESHOLD)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(
    source_video_path: str,
    device: str,
    high_accuracy: bool = False
) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        high_accuracy (bool): Use higher accuracy settings.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    setup_gpu_optimizations(device)
    player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    imgsz = DEFAULT_PLAYER_IMGSZ if high_accuracy else 1280

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(
            frame, imgsz=imgsz, conf=DEFAULT_PLAYER_CONF, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(
            frame, imgsz=imgsz, conf=DEFAULT_PLAYER_CONF, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=DEFAULT_NMS_THRESHOLD)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(
    source_video_path: str,
    device: str,
    high_accuracy: bool = False
) -> Iterator[np.ndarray]:
    """
    Run radar visualization with player and pitch detection.
    
    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on.
        high_accuracy (bool): Use higher accuracy settings.
        
    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    setup_gpu_optimizations(device)
    player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)
    pitch_detection_model = load_model(PITCH_DETECTION_MODEL_PATH, device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    imgsz = DEFAULT_PLAYER_IMGSZ if high_accuracy else 1280

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(
            frame, imgsz=imgsz, conf=DEFAULT_PLAYER_CONF, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(
            frame, imgsz=imgsz, conf=DEFAULT_PLAYER_CONF, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=DEFAULT_NMS_THRESHOLD)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def run_combined_detection(
    source_video_path: str,
    device: str,
    high_accuracy: bool = True
) -> Iterator[np.ndarray]:
    """
    Combined detection mode that does everything:
    - Player detection (high resolution)
    - Ball detection (high resolution)
    - Team classification
    - ByteTrack for both players and ball
    - Velocity-based ball prediction

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        high_accuracy (bool): Use high accuracy settings (default True).

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    setup_gpu_optimizations(device)
    
    # Load models with FP16
    player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)
    ball_detection_model = load_model(BALL_DETECTION_MODEL_PATH, device)
    pitch_detection_model = load_model(PITCH_DETECTION_MODEL_PATH, device)
    
    # Settings
    player_imgsz = DEFAULT_PLAYER_IMGSZ if high_accuracy else 1280
    ball_imgsz = DEFAULT_BALL_IMGSZ if high_accuracy else 640
    
    # First pass: collect crops for team classification
    print("[Combined] Collecting player crops for team classification...")
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)
    
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(
            frame, imgsz=player_imgsz, conf=DEFAULT_PLAYER_CONF, verbose=False
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
    
    # Train team classifier
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    
    # Initialize trackers
    player_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_bytetrack = sv.ByteTrack(
        minimum_consecutive_frames=3,
        lost_track_buffer=30,
    )
    
    # Enhanced ball tracker with velocity prediction
    ball_tracker = BallTracker(
        buffer_size=30,
        velocity_alpha=0.3,
        max_prediction_frames=5
    )
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    
    # Ball detection callback with overlapping slices
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(
            image_slice, imgsz=ball_imgsz, conf=DEFAULT_BALL_CONF, verbose=False
        )[0]
        return sv.Detections.from_ultralytics(result)
    
    ball_slicer = create_inference_slicer(ball_callback, slice_wh=(640, 640))
    
    # Second pass: full detection
    print("[Combined] Running full detection pipeline...")
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    for frame in frame_generator:
        # Pitch detection for radar
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        
        # Player detection
        player_result = player_detection_model(
            frame, imgsz=player_imgsz, conf=DEFAULT_PLAYER_CONF, verbose=False
        )[0]
        player_detections = sv.Detections.from_ultralytics(player_result)
        player_detections = player_detections.with_nms(threshold=DEFAULT_NMS_THRESHOLD)
        player_detections = player_tracker.update_with_detections(player_detections)
        
        # Ball detection with slicing
        ball_detections = ball_slicer(frame).with_nms(threshold=DEFAULT_NMS_THRESHOLD)
        ball_detections = ball_bytetrack.update_with_detections(ball_detections)
        ball_detections = ball_tracker.update(ball_detections)
        
        # Team classification
        players = player_detections[player_detections.class_id == PLAYER_CLASS_ID]
        player_crops = get_crops(frame, players)
        if len(player_crops) > 0:
            players_team_id = team_classifier.predict(player_crops)
        else:
            players_team_id = np.array([])
        
        goalkeepers = player_detections[player_detections.class_id == GOALKEEPER_CLASS_ID]
        if len(goalkeepers) > 0 and len(players) > 0 and len(players_team_id) > 0:
            goalkeepers_team_id = resolve_goalkeepers_team_id(
                players, players_team_id, goalkeepers)
        else:
            goalkeepers_team_id = np.array([])
        
        referees = player_detections[player_detections.class_id == REFEREE_CLASS_ID]
        
        # Merge all player detections
        all_player_detections = sv.Detections.merge([players, goalkeepers, referees])
        if len(all_player_detections) > 0:
            color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
            )
            labels = [str(tracker_id) for tracker_id in all_player_detections.tracker_id]
        else:
            color_lookup = np.array([])
            labels = []
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Draw players with team colors
        if len(all_player_detections) > 0:
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, all_player_detections, custom_color_lookup=color_lookup)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, all_player_detections, labels, custom_color_lookup=color_lookup)
        
        # Draw ball
        if len(ball_detections) > 0:
            annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
        
        # Draw radar overlay
        if len(all_player_detections) > 0:
            try:
                h, w, _ = frame.shape
                radar = render_radar(all_player_detections, keypoints, color_lookup)
                radar = sv.resize_image(radar, (w // 2, h // 2))
                radar_h, radar_w, _ = radar.shape
                rect = sv.Rect(
                    x=w // 2 - radar_w // 2,
                    y=h - radar_h,
                    width=radar_w,
                    height=radar_h
                )
                annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
            except Exception:
                pass  # Skip radar if transformation fails
        
        yield annotated_frame


def run_frame_extraction(
    source_video_path: str,
    output_dir: str,
    device: str,
    high_accuracy: bool = False
) -> None:
    """
    Extract all frames from video, run detection, and save annotated frames as images.
    
    Args:
        source_video_path: Path to the source video.
        output_dir: Directory to save annotated frames and metadata.
        device: Device to run the model on.
        high_accuracy: Use higher accuracy settings.
    
    Output structure:
        output_dir/
            frames/          - All annotated frame images
            metadata.json    - Detection data with timestamps
            detections.csv   - CSV with frame-by-frame detections
    """
    setup_gpu_optimizations(device)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Load models
    print("[Frame Extraction] Loading models...")
    player_detection_model = load_model(PLAYER_DETECTION_MODEL_PATH, device)
    ball_detection_model = load_model(BALL_DETECTION_MODEL_PATH, device)
    
    # Get video info
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps if video_info.fps else 30.0
    total_frames = video_info.total_frames
    
    print(f"[Frame Extraction] Video: {total_frames} frames @ {fps} FPS")
    print(f"[Frame Extraction] Duration: {total_frames / fps:.2f} seconds")
    print(f"[Frame Extraction] Output: {output_dir}")
    
    # Settings
    player_imgsz = DEFAULT_PLAYER_IMGSZ if high_accuracy else 1280
    ball_imgsz = DEFAULT_BALL_IMGSZ if high_accuracy else 640
    
    # Trackers
    player_tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=30, velocity_alpha=0.3, max_prediction_frames=5)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    
    # Ball slicer
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(
            image_slice, imgsz=ball_imgsz, conf=DEFAULT_BALL_CONF, verbose=False
        )[0]
        return sv.Detections.from_ultralytics(result)
    
    ball_slicer = create_inference_slicer(ball_callback, slice_wh=(640, 640))
    
    # Storage for metadata
    all_detections: List[Dict[str, Any]] = []
    
    # Process frames
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    print("[Frame Extraction] Processing frames...")
    for frame_idx, frame in enumerate(tqdm(frame_generator, total=total_frames, desc="Extracting")):
        timestamp_sec = frame_idx / fps
        timestamp_str = f"{int(timestamp_sec // 60):02d}:{timestamp_sec % 60:05.2f}"
        
        # Player detection
        player_result = player_detection_model(
            frame, imgsz=player_imgsz, conf=DEFAULT_PLAYER_CONF, verbose=False
        )[0]
        player_detections = sv.Detections.from_ultralytics(player_result)
        player_detections = player_detections.with_nms(threshold=DEFAULT_NMS_THRESHOLD)
        player_detections = player_tracker.update_with_detections(player_detections)
        
        # Ball detection
        ball_detections = ball_slicer(frame).with_nms(threshold=DEFAULT_NMS_THRESHOLD)
        ball_detections = ball_tracker.update(ball_detections)
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Add timestamp to frame
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_idx} | Time: {timestamp_str}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Draw player boxes
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, player_detections)
        labels = build_player_labels(player_detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
            annotated_frame, player_detections, labels=labels
        )
        
        # Draw ball
        if len(ball_detections) > 0:
            annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
        
        # Save frame
        frame_filename = f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(os.path.join(frames_dir, frame_filename), annotated_frame)
        
        # Collect detection data
        frame_data = {
            "frame_index": frame_idx,
            "timestamp_sec": round(timestamp_sec, 3),
            "timestamp_str": timestamp_str,
            "players": [],
            "ball": None
        }
        
        # Player data
        for i in range(len(player_detections)):
            bbox = player_detections.xyxy[i].tolist()
            tracker_id = int(player_detections.tracker_id[i]) if player_detections.tracker_id is not None else None
            class_id = int(player_detections.class_id[i]) if player_detections.class_id is not None else None
            conf = float(player_detections.confidence[i]) if player_detections.confidence is not None else None
            
            frame_data["players"].append({
                "tracker_id": tracker_id,
                "class": CLASS_NAME_LOOKUP.get(class_id, "unknown"),
                "class_id": class_id,
                "bbox": [round(x, 1) for x in bbox],
                "confidence": round(conf, 3) if conf else None
            })
        
        # Ball data
        if len(ball_detections) > 0:
            ball_bbox = ball_detections.xyxy[0].tolist()
            ball_center = ball_detections.get_anchors_coordinates(sv.Position.CENTER)[0].tolist()
            ball_conf = float(ball_detections.confidence[0]) if ball_detections.confidence is not None else None
            frame_data["ball"] = {
                "bbox": [round(x, 1) for x in ball_bbox],
                "center": [round(x, 1) for x in ball_center],
                "confidence": round(ball_conf, 3) if ball_conf else None
            }
        
        all_detections.append(frame_data)
    
    # Save metadata JSON
    metadata_path = os.path.join(output_dir, "metadata.json")
    print(f"[Frame Extraction] Saving metadata to {metadata_path}")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "video_info": {
                "source": source_video_path,
                "total_frames": total_frames,
                "fps": fps,
                "duration_sec": round(total_frames / fps, 2),
                "width": video_info.width,
                "height": video_info.height
            },
            "frames": all_detections
        }, f, indent=2)
    
    # Save CSV for easy viewing
    csv_path = os.path.join(output_dir, "detections.csv")
    print(f"[Frame Extraction] Saving CSV to {csv_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "timestamp", "num_players", "ball_detected",
            "ball_x", "ball_y", "player_ids"
        ])
        for det in all_detections:
            ball_x = det["ball"]["center"][0] if det["ball"] else ""
            ball_y = det["ball"]["center"][1] if det["ball"] else ""
            player_ids = ",".join(str(p["tracker_id"]) for p in det["players"] if p["tracker_id"])
            writer.writerow([
                det["frame_index"],
                det["timestamp_str"],
                len(det["players"]),
                "Yes" if det["ball"] else "No",
                ball_x,
                ball_y,
                player_ids
            ])
    
    print(f"\n✅ Done! Output saved to: {output_dir}")
    print(f"   - Frames: {frames_dir}/ ({total_frames} images)")
    print(f"   - Metadata: {metadata_path}")
    print(f"   - CSV: {csv_path}")


def main(
    source_video_path: str,
    target_video_path: str,
    device: str,
    mode: Mode,
    show_preview: bool = False,
    ball_output_dir: Optional[str] = None,
    pass_csv_path: Optional[str] = None,
    high_accuracy: bool = False,
    output_dir: Optional[str] = None
) -> None:
    if mode == Mode.FRAME_EXTRACTION:
        run_frame_extraction(
            source_video_path=source_video_path,
            output_dir=output_dir or os.path.join(PARENT_DIR, "extracted_frames"),
            device=device,
            high_accuracy=high_accuracy
        )
        return
    
    if mode == Mode.BALL_DETECTION_RECOVERY:
        run_ball_detection_with_recovery(
            source_video_path=source_video_path,
            target_video_path=target_video_path,
            device=device,
            output_dir=ball_output_dir or os.path.join(PARENT_DIR, "ball_outputs"),
            pass_csv_path=pass_csv_path,
            show_preview=show_preview,
            high_accuracy=high_accuracy
        )
        return

    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device, high_accuracy=high_accuracy)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device, high_accuracy=high_accuracy)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device, high_accuracy=high_accuracy)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device, high_accuracy=high_accuracy)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device, high_accuracy=high_accuracy)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device, high_accuracy=high_accuracy)
    elif mode == Mode.COMBINED_DETECTION:
        frame_generator = run_combined_detection(
            source_video_path=source_video_path, device=device, high_accuracy=high_accuracy)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            if show_preview:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    if show_preview:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Soccer AI - Player and Ball Detection')
    parser.add_argument('--source_video_path', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--target_video_path', type=str, default='output.mp4',
                       help='Path to output video file (not used for FRAME_EXTRACTION)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu, cuda)')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION,
                       help='Detection mode')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for FRAME_EXTRACTION mode')
    parser.add_argument('--ball_output_dir', type=str, default=None)
    parser.add_argument('--pass_csv_path', type=str, default=None)
    parser.add_argument('--show_preview', action='store_true')
    parser.add_argument('--high_accuracy', action='store_true',
                       help='Use high accuracy mode (higher resolution, better tracking)')
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        show_preview=args.show_preview,
        ball_output_dir=args.ball_output_dir,
        pass_csv_path=args.pass_csv_path,
        high_accuracy=args.high_accuracy,
        output_dir=args.output_dir
    )
