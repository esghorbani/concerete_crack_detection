import os
from io import BytesIO
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch


class CrackDetector:
    def __init__(self, model_path: str | None = None) -> None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        if model_path is None:
            model_path = os.path.join(base_dir, "src", "models", "best_model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def _get_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )

    def predict(self, image_pil: Image.Image, threshold: float = 0.5) -> dict[str, Any]:
        image_rgb = np.array(image_pil.convert("RGB"))
        original_h, original_w = image_rgb.shape[:2]

        transformed = self.transform(image=image_rgb)
        image_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        mask = (probs > threshold).astype(np.uint8) * 255
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        overlay = image_rgb.copy()
        overlay[mask > 0] = [255, 140, 0]
        blended = cv2.addWeighted(image_rgb, 0.72, overlay, 0.28, 0)

        crack_pixels = int(np.sum(mask > 0))
        total_pixels = mask.shape[0] * mask.shape[1]
        crack_ratio = (crack_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        crack_detected = crack_pixels > 0

        return {
            "crack_detected": crack_detected,
            "crack_pixels": crack_pixels,
            "crack_area_percent": crack_ratio,
            "mask_array": mask,
            "overlay_array": blended,
            "mask_pil": Image.fromarray(mask),
            "overlay_pil": Image.fromarray(blended),
        }

    @staticmethod
    def pil_to_png_bytes(pil_image: Image.Image) -> bytes:
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()