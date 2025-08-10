import cv2
from ultralytics import YOLO
import sys
import os
from src.Depth_estimation import DepthEstimator
from src.spatial_positions import calculate_spatial_position
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def object_detection(frame, label):
    """
    Detects the object of interest (OOI) in the given frame using YOLOv8n model.
    Args:
        frame (numpy.ndarray): The video frame to analyze.
        ooi_label (str): The label of the object to detect.
    Returns:
        tuple: (found (bool), detections (list of dicts with 'label' and 'bbox'))
    """
    # Load YOLOv8n model (make sure the model is downloaded only once)
    if not hasattr(object_detection, 'model'):
        object_detection.model = YOLO('yolov8n.pt')
    model = object_detection.model
    results = model(frame)
    detections = []
    target_found = False
    target_detection = None
    for result in results:
        boxes = result.boxes
        names = result.names
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            depth_estimator = DepthEstimator()
            # Estimate depth using MiDaS monocular depth estimation
            depth = depth_estimator.estimate_depth(frame, [x1, y1, x2, y2])

            # Calculate spatial position
            h_pos, v_pos, norm_x, norm_y = calculate_spatial_position([x1, y1, x2, y2], frame.shape)

            # Add detection to list
            detection = {
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'depth_feet': depth,
                'horizontal_position': h_pos,
                'vertical_position': v_pos,
                'normalized_x': norm_x,
                'normalized_y': norm_y
            }
            detections.append(detection)

            # Check if target object found
            if label.lower() in class_name.lower() and confidence > 0.5:
                    target_found = True
                    target_detection = detection

    return detections, target_found, target_detection
