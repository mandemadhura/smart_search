import cv2
from ultralytics import YOLO
import sys
import os
from datetime import datetime
from src.Depth_estimation import DepthEstimator
from src.spatial_positions import calculate_spatial_position
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def object_detection(frame, label):
    if not hasattr(object_detection, 'model'):
        object_detection.model = YOLO('yolo11x.pt')
    model = object_detection.model

    if not hasattr(object_detection, 'depth_estimator'):
        object_detection.depth_estimator = DepthEstimator(verbose=True)
    depth_estimator = object_detection.depth_estimator

    # Save folder setup (can comment out if not needed)
    save_folder = "detected_object_frames"
    os.makedirs(save_folder, exist_ok=True)

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
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if confidence < 0.2:
                continue

            if label.lower() in class_name.lower():
                depth = depth_estimator.estimate_depth(frame, [x1, y1, x2, y2], normalized=False)
                h_pos, v_pos, norm_x, norm_y = calculate_spatial_position([x1, y1, x2, y2], frame.shape)

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
                target_found = True
                target_detection = detection
                detections.append(detection)

                #----------- Save frame when target detected (comment/uncomment to enable/disable) -----------
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(save_folder, f"frame_{timestamp}.png")
                cv2.imwrite(filename, frame)
                print(f"[INFO] Saved frame to {filename}")
                #-------------------------------------------------------------------------------------------

            else:
                detection = {
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                }
                detections.append(detection)

    return detections, target_found, target_detection
