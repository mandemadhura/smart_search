import torch
import cv2
import numpy as np
from PIL import Image
import depth_pro

from src.models.depth_model import DepthModel


class MlDepthPro(DepthModel):
    """
    Implements ml depth pro model to get
    the depth of an object from an image
    """
    def __init__(self, device: str = None, model_type: str = None):
        """Base initialization"""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model_type=model_type, device=device)
        self.model, self.transformer = self.load_model(device=device)

    def load_model(self, device: str) -> tuple:
        """load and transform ml depth pro model"""
        print(f"🔧 Using {device.upper()} to load and run ML Depth Pro model...")

        try:
            # Load model and transforms
            model, transform = depth_pro.create_model_and_transforms()
            model.to(device)
            model.eval()
            print(f"✅ Depth Pro model loaded successfully on {device.upper()}.")
            return model, transform
        except (FileNotFoundError, EOFError, torch.serialization.pickle.UnpicklingError) as e:
            print(f"❌ Model load error: {type(e).__name__}: {e}")
            print("Try re-downloading or verifying the checkpoint path.")
            raise

    def preprocess_image(self, frame: np.ndarray) -> tuple:
        """
        Preprocess image to a specific format needed
        for model transformer. Also calculate focal length
        required to get the depth of an object from the image.
        """
        frame = cv2.resize(frame, (320, 320))
        # Convert frame (BGR → RGB → PIL)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        # Estimate focal length (in pixels)
        # Heuristic focal length (float → tensor)
        f_px = max(pil_img.size) * 1.2
        f_px_tensor = torch.tensor([f_px], dtype=torch.float32, device=self.device)
        # Preprocess and run model
        img_t = self.transformer(pil_img).unsqueeze(0).to(self.device)
        return img_t, f_px_tensor
        
    def estimate_depth(self, frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        """Get the predicted depth by inferencing the model"""
        try:
            print(f"Estimating the depth...")
            img_t, f_px_tensor = self.preprocess_image(frame=frame)
            with torch.no_grad():
                pred = self.model.infer(img_t, f_px=f_px_tensor)
            # Extract metric depth (meters)
            depth_m = pred["depth"]
            if isinstance(depth_m, torch.Tensor):
                depth_m = depth_m.detach().cpu().numpy()
            depth_m = np.squeeze(depth_m)  # remove batch/channel dimensions
            if depth_m.ndim != 2:
                print(f"Warning: depth map is not 2D, shape={depth_m.shape}, skipping frame")
            depth = MlDepthPro.calculate_median_depth(depth=depth_m, box=box)
            return depth
        except Exception as err:
            print(f"❌ Unexpected error during inference: {err}")
            raise

    @staticmethod
    def calculate_median_depth(depth: np.array, box: np.array):
        """
        Calculate the median depth of an object from the depth map
        and the bounding box of the object.
        """
        median_depth = 0
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        h, w = depth.shape
        # limit the width values of YOLO object box within range
        # 0 to its width
        x1, x2 = np.clip([x1, x2], 0, w - 1)
        # limit the height values of YOLO object box within range
        # 0 to its height
        y1, y2 = np.clip([y1, y2], 0, h - 1)
        # In numpy, Y is rows and X is column. Hence use
        # depth_m[y1:y2, x1:x2] depth_m[rows, column]
        # From full depth map, take the rectangular slice that
        # corresponds to the object((x1, y1), (x2, y2))
        obj_depth = depth[y1:y2, x1:x2]
        if obj_depth.size > 0:
            median_depth = np.nanmedian(obj_depth)
        print(f"Estimated object depth is: {median_depth} meters")
        depth = median_depth * 3.28084
        print(f"Estimated Object depth is:{depth} feet")
        return median_depth
