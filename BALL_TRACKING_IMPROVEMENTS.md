# Ball Detection & Tracking Improvements

## Answers to Your Questions

### 1. Does the model run on live videos?
**No**, the current implementation processes **pre-recorded video files** frame-by-frame. It uses `sv.get_video_frames_generator()` which extracts frames from video files, not live camera feeds.

### 2. What were the ball tracking issues?
The original implementation had several problems:
- **Simple tracking**: Only picked detection closest to centroid, no proper tracking algorithm
- **No prediction**: When ball wasn't detected, it just returned empty detections
- **Slicer boundaries**: Balls at slice boundaries could be missed
- **No velocity model**: Couldn't predict where ball should be during occlusions

## Improvements Made

### 1. Enhanced BallTracker (`sports/common/ball.py`)
- ✅ **Velocity-based prediction**: Predicts ball position when missing for up to 5 frames
- ✅ **Smoother velocity calculation**: Uses exponential moving average
- ✅ **Better buffer management**: Larger buffer (30 frames) for better prediction
- ✅ **Handles edge cases**: Better handling of empty detections and coordinate issues

### 2. Improved Ball Detection (`examples/soccer/main.py`)
- ✅ **ByteTrack integration**: Uses proper tracking algorithm (same as players) for consistency
- ✅ **Overlapping slices**: 20% overlap to catch balls at slice boundaries
- ✅ **Lower confidence threshold**: `conf=0.15` to catch more ball detections
- ✅ **Better NMS threshold**: `0.2` for better filtering
- ✅ **Dual tracking**: ByteTrack for consistency + velocity prediction for occlusions

### 3. Key Changes

#### Detection Parameters:
```python
# Lower confidence to catch more balls
conf=0.15  # (was default ~0.25)

# Overlapping slices
overlap_ratio_wh=(0.2, 0.2)  # 20% overlap

# Better NMS
with_nms(threshold=0.2)  # (was 0.1)
```

#### Tracking:
```python
# ByteTrack for consistency
ball_byte_tracker = sv.ByteTrack(
    track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    minimum_consecutive_frames=2
)

# Enhanced tracker for velocity prediction
ball_tracker = BallTracker(buffer_size=30)
```

## How It Works Now

1. **Frame Extraction**: Video is processed frame-by-frame (not live)
2. **Detection**: Ball detected using YOLO with overlapping slices
3. **ByteTrack**: Maintains consistent track IDs across frames
4. **Velocity Prediction**: If ByteTrack loses ball, velocity-based prediction fills gaps
5. **Visualization**: Ball trail shows recent positions with color gradient

## Testing Recommendations

1. **Compare before/after**: Run on same video with old vs new code
2. **Check missed frames**: Look for improvements in ball visibility
3. **Track consistency**: Verify track IDs remain stable
4. **Occlusion handling**: Test scenes where ball is briefly occluded

## Future Improvements (Optional)

- [ ] Kalman filter for better velocity prediction
- [ ] Multi-scale detection for different ball sizes
- [ ] Temporal smoothing of detections
- [ ] Integration with player positions for context
- [ ] Adaptive confidence thresholds based on scene

## Usage

The improvements are automatically applied when using:
```bash
python main.py --mode BALL_DETECTION --source_video_path <video> --target_video_path <output>
```

Or with recovery mode:
```bash
python main.py --mode BALL_DETECTION_RECOVERY --source_video_path <video> --target_video_path <output> --ball_output_dir <dir>
```

