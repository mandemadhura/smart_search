import time
from src.voice_to_text import voice_to_text
from src.label_detection import detect_object_label
from src.object_detection import ObjectDetector
from src.frame_freeze_of_ooi import find_frame_with_object
from src.spatial_context import get_spatial_context_description
from src.text_to_speech import text_to_speech
from src.config.config import load_config, get_depth_model
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    start_time = time.time()

    # 1. Get voice input and convert to text
    
    speak_out= " Hello there, please tell me what you are looking for."
    text_to_speech(speak_out)
    print(speak_out)
    
    t1 = time.time()
    text = voice_to_text()
    t2 = time.time()
    if not text:
        print("No voice input detected.")
        return
    print(f"[Time] Voice input and conversion took {t2 - t1:.2f} seconds")

    # 2. Detect object of interest (OOI) label from text
    t3 = time.time()
    label = detect_object_label(text)
    t4 = time.time()
    if not label:
        print("No object of interest detected in the input.")
        return
    print(f"[Time] Label detection took {t4 - t3:.2f} seconds")

    print(f"Object of Interest: {label}")
    label_speak_out = f"The object you are looking for is {label}."
    text_to_speech(label_speak_out)

    config = load_config(config_path="./config.yaml")
    depth_model = get_depth_model(config=config)
    object_detector_model = config.get('object_detection_model').get('model', 'yolov8n.pt')
    object_detector = ObjectDetector(model=object_detector_model, depth_model=depth_model)
    # 3. Find frame with OOI and get detections
    t5 = time.time()
    frame, found_detections, target_detection = \
        find_frame_with_object(
                    label=label,
                    object_detector=object_detector
        )

    t6 = time.time()
    if frame is None or found_detections is None:
        print(f"No frame with '{label}' found.")
        return
    print(f"[Time] Frame finding and object detection took {t6 - t5:.2f} seconds")

    found_object_speak_out = f"Found the object {label} in the scene."
    text_to_speech(found_object_speak_out)
    speak_out_calling_gpt = "Now, let me describe the spatial context of the scene. Give me a moment"
    text_to_speech(speak_out_calling_gpt)

    # 4. Get spatial context description using GPT-4o
    t7 = time.time()
    text2 = get_spatial_context_description(frame, found_detections, target_detection, label)
    t8 = time.time()
    print(f"[Time] Spatial context generation took {t8 - t7:.2f} seconds")

    print(f"Spatial Context Description: {text2}")

    # 5. Convert the description to speech
    text_to_speech(text2)
    # 6. Calculate total time taken t2-t1(voice to text) + t4-t3(label detection) + t6-t5(frame freeze and object detection) + t8-t7(spatial context generation)
    total_time = (t2-t1) + (t4-t3) + (t6-t5) + (t8-t7)
    end_time = time.time()
    print("\nTime Report:")
    print("------------")
    print(f"Voice-to-Text: {t2-t1:.2f} seconds")
    print(f"Label Detection: {t4-t3:.2f} seconds")
    print(f"Frame Freeze and Object Detection: {t6-t5:.2f} seconds")
    print(f"Spatial Context Generation: {t8-t7:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")
    print("Description spoken successfully.")
    print(f"[Total Time] Overall process took {total_time:.2f} seconds")
    print(f"[Total Time] Including overhead, total time took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
