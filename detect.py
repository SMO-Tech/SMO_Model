import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

class FootballDetector:
    def __init__(self, model='yolov8n.pt'):
        self.model = YOLO(model)

    def detect(self, video_path, max_frames=100):
        cap = cv2.VideoCapture(video_path)
        results = []

        for frame_num in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            dets = self.model(frame, verbose=False)

            for det in dets:
                if det.boxes:
                    for box in det.boxes:
                        if int(box.cls[0]) == 32:  # Ball
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            results.append({
                                'frame': frame_num,
                                'x': cx,
                                'y': cy,
                                'confidence': conf
                            })

        cap.release()
        return pd.DataFrame(results)

if __name__ == '__main__':
    detector = FootballDetector()
    df = detector.detect('input.mp4')
    df.to_csv('detections.csv', index=False)
    print(f'Found {len(df)} balls')