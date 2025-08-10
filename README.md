# Smart Search: AI-Powered Object detection for the Visually Impaired

## Overview
Smart Search is an end-to-end assistive AI system designed to help visually impaired users locate and navigate to objects of interest in their environment using voice commands, computer vision, spatial reasoning, and natural language guidance.

## Features
- **Voice-to-Text:** Users speak their request (e.g., "Where is my bottle?") and the system transcribes it to text.
- **Object of Interest (OOI) Detection:** The system extracts the target object label from the user's speech.
- **Live Video Feed & Object Detection:** The webcam scans the environment, using YOLOv8 for real-time object detection.
- **Depth Estimation:** For each detected object, the system estimates its distance using MiDaS monocular depth estimation.
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
src/
├── config/                # Configuration and environment variables
│   └── config.py
├── voice_to_text.py       # Voice input and speech recognition
├── ooi_detection.py       # Object of interest extraction from text
├── object_detection.py    # YOLOv8 object detection and spatial reasoning
├── Depth_estimation.py    # MiDaS depth estimation
├── spatial_positions.py   # Calculate left/right/center, etc.
├── frame_freeze_of_ooi.py # Freeze and save frame with OOI
├── spatial_context.py     # Generate spatial context and navigation with GPT-4o
├── text_to_speech.py      # Text-to-speech output
└── 
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
4. **Run the main program:**
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

## Acknowledgements
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [MiDaS Depth Estimation](https://github.com/isl-org/MiDaS)
- [OpenAI GPT-4o](https://platform.openai.com/docs/models/gpt-4o)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
