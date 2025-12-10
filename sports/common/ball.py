from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    """
    A class to annotate frames with circles of varying radii and colors.

    Attributes:
        radius (int): The maximum radius of the circles to be drawn.
        buffer (deque): A deque buffer to store recent coordinates for annotation.
        color_palette (sv.ColorPalette): A color palette for the circles.
        thickness (int): The thickness of the circle borders.
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):

        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolates the radius between 1 and the maximum radius based on the index.

        Args:
            i (int): The current index in the buffer.
            max_i (int): The maximum index in the buffer.

        Returns:
            int: The interpolated radius.
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotates the frame with circles based on detections.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections containing coordinates.

        Returns:
            np.ndarray: The annotated frame.
        """
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    """
    Enhanced ball tracker with velocity-based prediction for handling occlusions.

    Features:
    - Velocity-based prediction when ball is missing (up to max_prediction_frames)
    - Exponential moving average for smoother velocity calculation
    - Larger buffer (30 frames) for better prediction
    - Synthetic detection generation during occlusions

    Attributes:
        buffer (deque): A deque buffer to store recent ball positions.
        velocity (np.ndarray): Current estimated velocity (exponential moving average).
        velocity_alpha (float): Smoothing factor for velocity EMA (0-1).
        last_position (np.ndarray): Last known ball position.
        frames_since_detection (int): Counter for frames without detection.
        max_prediction_frames (int): Maximum frames to predict without detection.
    """

    def __init__(
        self,
        buffer_size: int = 30,
        velocity_alpha: float = 0.3,
        max_prediction_frames: int = 5
    ):
        """
        Initialize the enhanced BallTracker.

        Args:
            buffer_size (int): Size of position history buffer. Default 30.
            velocity_alpha (float): Smoothing factor for velocity EMA (0-1).
                Higher = more responsive, lower = smoother. Default 0.3.
            max_prediction_frames (int): Max frames to predict ball position
                when detection is missing. Default 5.
        """
        self.buffer = deque(maxlen=buffer_size)
        self.velocity: Optional[np.ndarray] = None
        self.velocity_alpha = velocity_alpha
        self.last_position: Optional[np.ndarray] = None
        self.frames_since_detection = 0
        self.max_prediction_frames = max_prediction_frames
        self._last_bbox_size: Optional[Tuple[float, float]] = None

    def _update_velocity(self, new_position: np.ndarray) -> None:
        """
        Update velocity using exponential moving average.

        Args:
            new_position (np.ndarray): New ball position [x, y].
        """
        if self.last_position is not None:
            instant_velocity = new_position - self.last_position
            if self.velocity is None:
                self.velocity = instant_velocity
            else:
                # Exponential moving average for smoother velocity
                self.velocity = (
                    self.velocity_alpha * instant_velocity +
                    (1 - self.velocity_alpha) * self.velocity
                )
        self.last_position = new_position.copy()

    def _predict_position(self) -> Optional[np.ndarray]:
        """
        Predict ball position based on velocity when detection is missing.

        Returns:
            Optional[np.ndarray]: Predicted position or None if cannot predict.
        """
        if self.last_position is None or self.velocity is None:
            return None
        if self.frames_since_detection >= self.max_prediction_frames:
            return None

        # Predict using velocity with slight decay for uncertainty
        decay = 0.9 ** self.frames_since_detection
        predicted = self.last_position + self.velocity * decay
        return predicted

    def _create_synthetic_detection(
        self,
        position: np.ndarray,
        confidence: float = 0.5
    ) -> sv.Detections:
        """
        Create a synthetic detection at the predicted position.

        Args:
            position (np.ndarray): Predicted position [x, y].
            confidence (float): Confidence score for synthetic detection.

        Returns:
            sv.Detections: Synthetic detection object.
        """
        # Use last known bbox size or default
        if self._last_bbox_size is not None:
            w, h = self._last_bbox_size
        else:
            w, h = 20.0, 20.0  # Default ball size

        x, y = position
        xyxy = np.array([[x - w/2, y - h/2, x + w/2, y + h/2]], dtype=np.float32)

        return sv.Detections(
            xyxy=xyxy,
            confidence=np.array([confidence], dtype=np.float32),
            class_id=np.array([0], dtype=np.int64),  # Ball class ID
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the tracker with new detections and returns the best detection.

        When ball is detected: selects detection closest to centroid of recent positions.
        When ball is missing: uses velocity-based prediction for up to max_prediction_frames.

        Args:
            detections (sv.Detections): The current frame's ball detections.

        Returns:
            sv.Detections: The best detection (real or synthetic if predicting).
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)

        if len(detections) == 0:
            # No detection - try to predict
            self.frames_since_detection += 1
            predicted_pos = self._predict_position()

            if predicted_pos is not None:
                # Update last position with prediction for next frame
                self.last_position = predicted_pos
                # Create synthetic detection
                synthetic = self._create_synthetic_detection(
                    predicted_pos,
                    confidence=max(0.1, 0.5 - 0.1 * self.frames_since_detection)
                )
                # Add to buffer for continuity
                self.buffer.append(predicted_pos.reshape(1, 2))
                return synthetic

            # Cannot predict - return empty
            self.buffer.append(xy)  # Empty array
            return detections

        # We have detections - reset prediction counter
        self.frames_since_detection = 0

        # Store bbox size for synthetic detections
        if len(detections) > 0:
            bbox = detections.xyxy[0]
            self._last_bbox_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

        # Select best detection based on centroid proximity
        if len(self.buffer) > 0 and any(len(b) > 0 for b in self.buffer):
            # Get all non-empty positions from buffer
            valid_positions = [b for b in self.buffer if len(b) > 0]
            if valid_positions:
                centroid = np.mean(np.concatenate(valid_positions), axis=0)
                distances = np.linalg.norm(xy - centroid, axis=1)
                index = np.argmin(distances)
            else:
                index = 0
        else:
            # No history - pick highest confidence or first
            if detections.confidence is not None and len(detections.confidence) > 0:
                index = np.argmax(detections.confidence)
            else:
                index = 0

        best_detection = detections[[index]]
        best_position = xy[index]

        # Update velocity and buffer
        self._update_velocity(best_position)
        self.buffer.append(best_position.reshape(1, 2))

        return best_detection

    def reset(self) -> None:
        """Reset the tracker state."""
        self.buffer.clear()
        self.velocity = None
        self.last_position = None
        self.frames_since_detection = 0
        self._last_bbox_size = None
