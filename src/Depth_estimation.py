# src/Depth_estimation.py  (replace relevant parts)
import cv2
import numpy as np
import torch
import math
from typing import List, Tuple, Optional

class DepthEstimator:
    def __init__(self, model_type="MiDaS_small", device: Optional[str] = None, verbose: bool = True):
        self.verbose = verbose
        self.model_type = model_type
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.transform = None

        # Calibration params (optional)
        self.calib_a = None
        self.calib_b = None
        self.calib_eps = 1e-6

        # ROI selection params
        self.inner_crop_ratio = 0.8
        self.use_center_patch = False
        self.center_patch_ratio = 0.25

        try:
            # Attempt load
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.to(self.device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            if self.verbose:
                print(f"[DepthEstimator] Loaded {model_type} on {self.device}")
        except Exception as e:
            print(f"[DepthEstimator] Could not load MiDaS model: {e}")
            self.model = None
            self.transform = None

    def _bbox_to_pixels(self, bbox: List[float], image_shape: Tuple[int, int], normalized: bool):
        h, w = image_shape[:2]
        if normalized:
            x1 = int(round(bbox[0] * w))
            y1 = int(round(bbox[1] * h))
            x2 = int(round(bbox[2] * w))
            y2 = int(round(bbox[3] * h))
        else:
            x1, y1, x2, y2 = map(int, bbox)
        return [x1, y1, x2, y2]

    def _clamp_and_fix_bbox(self, bbox_px: List[int], image_shape: Tuple[int, int], min_size_px: int = 5):
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox_px
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        if x2 <= x1:
            x2 = min(w, x1 + min_size_px)
            x1 = max(0, x2 - min_size_px)
        if y2 <= y1:
            y2 = min(h, y1 + min_size_px)
            y1 = max(0, y2 - min_size_px)

        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(x1 + 1, min(w, x2)))
        y2 = int(max(y1 + 1, min(h, y2)))

        return [x1, y1, x2, y2]

    def estimate_depth(self, image: np.ndarray, bbox: List[float], normalized: bool = False,
                       use_percentile: Optional[float] = None) -> float:
        bbox_px = self._bbox_to_pixels(bbox, image.shape, normalized)
        bbox_px = self._clamp_and_fix_bbox(bbox_px, image.shape)

        # Early check: model present
        if self.model is None or self.transform is None:
            if self.verbose:
                print("[DepthEstimator] No model/transform -> fallback")
            return self.estimate_depth_fallback(bbox_px, image.shape)

        x1, y1, x2, y2 = bbox_px

        # BGR -> RGB
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb_image = image

        # Apply transform safely
        try:
            transformed = self.transform(rgb_image)
            # transform might return a tensor (C,H,W) or dict; handle both
            if isinstance(transformed, dict):
                # many MiDaS transforms return just a tensor, but if a dict is returned, try common keys
                tensor = None
                for k in ("image", "input", "rgb"):
                    if k in transformed:
                        tensor = transformed[k]
                        break
                if tensor is None:
                    # try to grab the first tensor-like item
                    for v in transformed.values():
                        if torch.is_tensor(v):
                            tensor = v
                            break
                if tensor is None:
                    raise RuntimeError("Transform returned dict with no tensor")
                input_batch = tensor.to(self.device)
            else:
                input_batch = transformed.to(self.device)

        except Exception as e:
            if self.verbose:
                print(f"[DepthEstimator] transform failed: {e} -> fallback")
            return self.estimate_depth_fallback(bbox_px, image.shape)

        # Ensure batch dim
        if input_batch.dim() == 3:
            input_batch = input_batch.unsqueeze(0)

        # Model inference
        try:
            with torch.no_grad():
                prediction = self.model(input_batch)

                # Normalize common outputs: if model returns tuple/list, pick first
                if isinstance(prediction, (list, tuple)):
                    prediction = prediction[0]

                # If prediction has channel dim, try to unify
                if prediction.dim() == 4:
                    pred = prediction
                elif prediction.dim() == 3:
                    pred = prediction.unsqueeze(1)
                else:
                    pred = prediction

                up = torch.nn.functional.interpolate(
                    pred,
                    size=rgb_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                depth_map = up[0, 0].cpu().numpy()
        except Exception as e:
            if self.verbose:
                print(f"[DepthEstimator] model inference failed: {e} -> fallback")
            return self.estimate_depth_fallback(bbox_px, image.shape)

        # ROI
        # Ensure bbox still valid after conversions
        x1, y1, x2, y2 = [max(0, int(v)) for v in (x1, y1, x2, y2)]
        if x2 <= x1 or y2 <= y1:
            if self.verbose:
                print("[DepthEstimator] invalid bbox after clamping -> fallback")
            return self.estimate_depth_fallback(bbox_px, image.shape)

        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            if self.verbose:
                print("[DepthEstimator] empty ROI -> fallback")
            return self.estimate_depth_fallback(bbox_px, image.shape)

        # Inner crop
        if self.inner_crop_ratio < 1.0:
            ih = int(max(1, (y2 - y1) * (1 - self.inner_crop_ratio) / 2.0))
            iw = int(max(1, (x2 - x1) * (1 - self.inner_crop_ratio) / 2.0))
            sy = ih
            ey = roi.shape[0] - ih
            sx = iw
            ex = roi.shape[1] - iw
            if ey > sy and ex > sx:
                roi = roi[sy:ey, sx:ex]

        if self.use_center_patch:
            ch = max(1, int(roi.shape[0] * self.center_patch_ratio))
            cw = max(1, int(roi.shape[1] * self.center_patch_ratio))
            cy = roi.shape[0] // 2
            cx = roi.shape[1] // 2
            y0 = max(0, cy - ch // 2)
            y1p = min(roi.shape[0], y0 + ch)
            x0 = max(0, cx - cw // 2)
            x1p = min(roi.shape[1], x0 + cw)
            roi = roi[y0:y1p, x0:x1p]

        roi = roi[np.isfinite(roi)]
        if roi.size == 0:
            if self.verbose:
                print("[DepthEstimator] ROI has no finite values -> fallback")
            return self.estimate_depth_fallback(bbox_px, image.shape)

        if use_percentile is not None:
            stat_val = float(np.percentile(roi, use_percentile))
        else:
            stat_val = float(np.median(roi))

        median_depth = stat_val
        if median_depth <= 0 or math.isnan(median_depth):
            if self.verbose:
                print(f"[DepthEstimator] bad median_depth={median_depth} -> fallback")
            return self.estimate_depth_fallback(bbox_px, image.shape)

        # Use calibration if present; else heuristic
        if self.calib_a is not None and self.calib_b is not None:
            distance_feet = float(self.calib_a / (median_depth + self.calib_eps) + self.calib_b)
            distance_feet = float(np.clip(distance_feet, 0.5, 200.0))
            return round(distance_feet, 2)

        # heuristic fallback mapping (non-calibrated)
        fallback_a = 8.0
        fallback_b = 0.5
        distance_feet = fallback_a / (median_depth + 1e-6) + fallback_b
        distance_feet = float(np.clip(distance_feet, 0.5, 200.0))
        if self.verbose:
            print("[DepthEstimator] Warning: no calibration — heuristic result")
        return round(distance_feet, 2)

    @staticmethod
    def estimate_depth_fallback(bbox_px: List[int], image_shape: Tuple[int, int]) -> float:
        height, width = image_shape[:2]
        x1, y1, x2, y2 = bbox_px
        object_height = max(1, y2 - y1)
        relative_height = object_height / height
        if relative_height > 0.45:
            return 3.0
        elif relative_height > 0.25:
            return 6.0
        elif relative_height > 0.12:
            return 10.0
        else:
            return 15.0
