# Frame Freeze and GPT Integration Module
# This module will freeze the frame and pass it to GPT for processing.

import cv2
import base64
from src.object_detection import ObjectDetector
import time

def find_frame_with_object(
          label: str,
          object_detector: ObjectDetector
):
    """
    Uses live video feed to find and return a frame containing the OOI, its base64 encoding, and detections.
    Processes only every `frame_skip`th frame.
    
    Args:
        label (str): The object of interest to look for.
        object_detection (callable): Function that takes (frame, label) and returns (detections, target_found, target_detection).
        frame_skip (int): Process every nth frame.
        
    Returns:
        tuple: (frame_base64 (str), detections (list), target_detection (dict or None))
    """
    cap = cv2.VideoCapture(0)
    found_frame_base64 = None
    found_detections = None
    target_detection = None
    frame_count = 0

    print(f"Looking for '{label}' in live video feed. Press 'q' to quit.")
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Live Feed', frame)
       
        t0 = time.time()
        if frame_count % 3 == 0:
            found_flag, box = object_detector.detect_object(frame=frame, label=label)
            t1 = time.time()
        frame_count += 1
        if found_flag:
            cap.release()
            cv2.destroyAllWindows()
            depth_estimation_start = time.time()
            detections, target_detection = \
                object_detector.create_response(frame=frame, box=box)
            depth_estimation_end = time.time()
            # frame_id = 0
            # if target_found:
            #     print(f"object_detection function took [Frame {frame_id}] OOI {label} DETECTED in {t1 - t0:.2f} seconds")
            # else:
            #     print(f"object_detection function took [Frame {frame_id}] OOI {label} NOT DETECTED in {t1 - t0:.2f} seconds")
            # frame_id += 1

            #print(f"Found '{label}' in frame!")
            print(f"[Time] object_detection function took [Frame {label} DETECTED in {t1 - t0:.2f} seconds")
            print(f"[Time] depth estimation function took {depth_estimation_end - depth_estimation_start:.2f} seconds")
            end_time = time.time()
            print(f"[Time] Frame freeze took with OOI(time spent in walking or searching) {end_time - start_time:.2f} seconds")
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', frame)
            found_frame_base64 = base64.b64encode(buffer).decode('utf-8')
            found_detections = detections
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return found_frame_base64, found_detections, target_detection

