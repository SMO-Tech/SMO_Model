# Football Ball Detection

## Real Results: 0.7% Detection Rate

### Your Actual Results:
- Detection rate: 0.7% (2 detections in 300 frames)
- Ball positions: Frame 260 (1309, 433), Frame 265 (1365, 487)
- Confidence: 0.393, 0.357

### Quick Start:
```bash
pip install -r requirements.txt
python detect.py --video football.mp4
```

### Files:
1. detect.py - Main code
2. highlights_detection_scenes.csv - Your results
3. highlights_detection.mp4 - Output video
4. ball_positions.csv - All positions

### Sample Data:
| Frame | X | Y | Confidence |
|-------|---|---|------------|
| 260 | 1309 | 433 | 0.393 |
| 265 | 1365 | 487 | 0.357 |