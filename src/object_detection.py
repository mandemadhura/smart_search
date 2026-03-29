from typing import Type
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
from datetime import datetime
from src.models.depth_model import DepthModel

from src.spatial_positions import calculate_spatial_position
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


class ObjectDetector:
    """Detects objet using YOLO model"""

    def __init__(self, model, depth_model: Type[DepthModel]):
        print(f"Loading {model} for object detection...")
        self.model = YOLO(model)
        self.depth_estimator = depth_model.estimate_depth

    def detect_object(self, frame: np.ndarray, label:str):
        results = self.model(frame)

        for result in results:
            boxes = result.boxes
            names = result.names

            for box in boxes:
                self.class_id = int(box.cls[0])
                self.class_name = names[self.class_id]
                self.confidence = float(box.conf[0])
                self.yolo_bounding_box = box.xyxy[0].tolist()

                if self.confidence < 0.2:
                    continue

                if label.lower() in self.class_name.lower():
                    return True, box
        return False, None

    def create_response(self, frame: np.ndarray, box: np.ndarray):
        detection = {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.yolo_bounding_box
        }
        detections = []
        target_found = False
        target_detection = None

        # Save folder setup (can comment out if not needed)
        save_folder = "detected_object_frames"
        os.makedirs(save_folder, exist_ok=True)

        depth = self.depth_estimator(frame=frame, box=box)
        h_pos, v_pos, norm_x, norm_y = \
                calculate_spatial_position(
                    self.yolo_bounding_box,
                    frame.shape
                )
        detection = {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.yolo_bounding_box,
            'depth_feet': depth,
            'horizontal_position': h_pos,
            'vertical_position': v_pos,
            'normalized_x': norm_x,
            'normalized_y': norm_y
        }
        target_found = True
        target_detection = detection
        detections.append(detection)
        #----------- Save frame when target detected (comment/uncomment to enable/disable) -----------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_folder, f"frame_{timestamp}.png")
        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved frame to {filename}")
        #-------------------------------------------------------------------------------------------

        return detections, target_detection