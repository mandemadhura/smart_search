# Smart Search: AI-Powered Object Navigation for the Visually Impaired

## Overview
Smart Search is an end-to-end assistive AI system designed to help visually impaired users locate and navigate to objects of interest in their environment using voice commands, computer vision, spatial reasoning, and natural language guidance.

## Features
- **Voice-to-Text:** Users speak their request (e.g., "Where is my bottle?") and the system transcribes it to text.
- **Object of Interest (OOI) Detection:** The system extracts the target object label from the user's speech.
- **Live Video Feed & Object Detection:** The webcam scans the environment, using YOLOv8 for real-time object detection.
- **Depth Estimation:** For each detected object, the system estimates its distance using configured monocular depth estimation(Either MiDaS or ml-depth-pro).
- **Spatial Reasoning:** Calculates the spatial position (left, right, center, etc.) and distance of each object.
- **Frame Freezing:** When the target object is found, the relevant video frame is frozen and saved.
- **Spatial Context with GPT-4o:** The system generates a clear, accessible spatial description and navigation guidance using GPT-4o, focusing on the target object and obstacles.
- **Text-to-Speech:** The spatial description and navigation instructions are spoken aloud to the user.

## How It Works
1. **User speaks a command** (e.g., "Find my bottle").
2. **Speech is transcribed** to text.
3. **Target object label** is extracted from the text.
4. **Webcam scans the scene**; YOLOv8 detects all objects in real time.
5. **Depth and spatial position** are calculated for each object.
6. **When the target is found**, the frame is frozen and saved.
7. **Spatial context and navigation instructions** are generated using GPT-4o, with a focus on the target and obstacles.
8. **Instructions are spoken** to the user.

## Project Structure
```
smart_search/
├── init.sh                      # Download ml-depth-pro model
├── checkpoints/                 # path for downloaded model .pt file
├── config.yaml                  # config file
├── src/
      ├── config
      │   └── config.py          # Configuration and environment variables
      ├── frame_freeze_of_ooi.py # Freeze and save frame with OOI
      ├── label_detection.py     # Object of interest extraction from text
      ├── models
      │   ├── depth_model.py
      │   ├── midas.py
      │   └── ml_depth_pro.py    # ml-depth-pro model for depth estimation
      ├── object_detection.py    # YOLOv8 object detection and spatial reasoning
      ├── spatial_context.py     # Generate spatial context and navigation with GPT-4o
      ├── spatial_positions.py   # Calculate left/right/center, etc.
      ├── text_to_speech.py      # Text-to-speech output
      └── voice_to_text.py       # Voice input and speech recognition
```

## Setup & Installation
1. **Clone the repository**
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up your `.env` file** with your Azure/OpenAI credentials:
   ```env
   ENDPOINT=your_azure_openai_endpoint
   API_KEY=your_azure_openai_api_key
   MODEL_NAME=gpt-4o
   ```
4. **Install ml-depth-pro model:**
   ```bash init.sh```

5. **Update config.yaml**
   - Update `name`, `device` and `params` parameter in yaml
     - For `depth_model` section:
         - Use name as `midas` for MiDaS model
         - Use name as `ml_depth_pro` for ml-depth-pro model
         - Use `cpu` or `cuda` for device
         - Update exact model type variant name. Ex. `MiDaS_small`
     - For `object_detection_model` section:
         - update `model` field with exact yolo model name to be used

6. **Run the main program:**
   ```sh
   python main.py
   ```

## Requirements
- Python 3.8+
- torch
- numpy
- opencv-python
- ultralytics
- timm
- python-dotenv
- azure-ai-inference
- openai
- depth_pro

## Acknowledgements
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [MiDaS Depth Estimation](https://github.com/isl-org/MiDaS)
- [OpenAI GPT-4o](https://platform.openai.com/docs/models/gpt-4o)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [ml-depth-pro Depth Estimation](https://github.com/apple/ml-depth-pro)