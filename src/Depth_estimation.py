import cv2
import numpy as np
import torch
import math

class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        """
        Load MiDaS depth estimation model once and prepare transforms
        model_type: "MiDaS_small" (fast), "DPT_Hybrid" (balanced), "DPT_Large" (most accurate)
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model and transforms
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.eval().to(self.device)

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

        except Exception as e:
            print(f"Could not load MiDaS: {e}")
            self.model = None
            self.transform = None

    def estimate_depth(self, image, bbox, normalized=False):
        """
        Estimate depth in feet from bounding box
        bbox: [x1, y1, x2, y2] in pixels OR normalized (if normalized=True)
        """
        if self.model is None:
            return self.estimate_depth_fallback(bbox, image.shape)

        height, width = image.shape[:2]

        # Convert normalized bbox to pixel coords if needed
        if normalized:
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
        else:
            x1, y1, x2, y2 = map(int, bbox)

        # Clamp values
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        # Skip if bbox too small or invalid
        if (x2 - x1) < 5 or (y2 - y1) < 5 or x2 <= x1 or y2 <= y1:
            return self.estimate_depth_fallback(bbox, image.shape)

        try:
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Transform & send to device
            input_batch = self.transform(rgb_image).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Convert to numpy
            depth_map = prediction.cpu().numpy()

            # Get depth ROI
            roi_depth = depth_map[y1:y2, x1:x2]
            if roi_depth.size == 0:
                return self.estimate_depth_fallback(bbox, image.shape)

            median_depth = np.median(roi_depth)

            # Convert disparity to feet
            if median_depth > 0:
                distance_feet = 1.5 + (math.log(median_depth + 1) * 2.5)
                distance_feet = max(1.0, min(25.0, distance_feet))
            else:
                distance_feet = 15.0

            return round(distance_feet, 1)

        except Exception as e:
            print(f"Depth estimation failed: {e}")
            return self.estimate_depth_fallback(bbox, image.shape)

    @staticmethod
    def estimate_depth_fallback(bbox, image_shape):
        """
        Fallback estimation using object size in frame
        """
        height, width = image_shape[:2]
        x1, y1, x2, y2 = bbox
        object_height = y2 - y1
        relative_height = object_height / height

        if relative_height > 0.4:
            return 3.0
        elif relative_height > 0.2:
            return 6.0
        elif relative_height > 0.1:
            return 10.0
        else:
            return 15.0
           