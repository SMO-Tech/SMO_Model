# How to Run main.py

The `main.py` script performs soccer video analysis with different modes.

## Basic Usage

```bash
python main.py \
    --source_video_path <input_video> \
    --target_video_path <output_video> \
    --device <device> \
    --mode <mode>
```

## Required Arguments

- `--source_video_path`: Path to input video file (e.g., `input_video/test_video.mp4`)
- `--target_video_path`: Path where output video will be saved (e.g., `output/result.mp4`)

## Optional Arguments

- `--device`: Device to run models on (`cpu`, `cuda`, `mps`). Default: `cpu`
- `--mode`: Analysis mode. Default: `PLAYER_DETECTION`

## Available Modes

1. **PITCH_DETECTION** - Detects and annotates pitch keypoints
2. **PLAYER_DETECTION** - Detects players with bounding boxes
3. **BALL_DETECTION** - Detects and tracks the ball
4. **PLAYER_TRACKING** - Tracks players across frames with IDs
5. **TEAM_CLASSIFICATION** - Classifies players into teams with colors
6. **RADAR** - Shows player positions on a radar view overlay

## Examples

### 1. Player Detection (Default)
```bash
python main.py \
    --source_video_path input_video/test_video.mp4 \
    --target_video_path output/player_detection.mp4 \
    --device cuda
```

### 2. Ball Detection
```bash
python main.py \
    --source_video_path input_video/test_video.mp4 \
    --target_video_path output/ball_detection.mp4 \
    --device cuda \
    --mode BALL_DETECTION
```

### 3. Player Tracking
```bash
python main.py \
    --source_video_path input_video/test_video.mp4 \
    --target_video_path output/player_tracking.mp4 \
    --device cuda \
    --mode PLAYER_TRACKING
```

### 4. Team Classification
```bash
python main.py \
    --source_video_path input_video/test_video.mp4 \
    --target_video_path output/team_classification.mp4 \
    --device cuda \
    --mode TEAM_CLASSIFICATION
```

### 5. Radar View
```bash
python main.py \
    --source_video_path input_video/test_video.mp4 \
    --target_video_path output/radar.mp4 \
    --device cuda \
    --mode RADAR
```

### 6. Pitch Detection
```bash
python main.py \
    --source_video_path input_video/test_video.mp4 \
    --target_video_path output/pitch_detection.mp4 \
    --device cuda \
    --mode PITCH_DETECTION
```

## Using CPU (if no GPU available)
```bash
python main.py \
    --source_video_path input_video/test_video.mp4 \
    --target_video_path output/result.mp4 \
    --device cpu \
    --mode PLAYER_DETECTION
```

## Notes

- The script will display a window showing the processed frames. Press 'q' to quit early.
- Make sure the model files exist in the `data/` directory:
  - `football-player-detection.pt`
  - `football-ball-detection.pt`
  - `football-pitch-detection.pt`
- For GPU acceleration, use `--device cuda` (requires CUDA-compatible GPU and PyTorch with CUDA support)
- Processing time depends on video length and device (GPU is much faster)

