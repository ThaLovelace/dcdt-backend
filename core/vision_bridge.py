"""
core/vision_bridge.py
---------------------
Digital Bridge: transforms a raw base-64 clock-drawing image into the
exact tensor format expected by ViT-B/16, while preserving the natural
sharpness of digital ink for clinical display.

Pipeline (Enhanced Resolution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  1. Base64 decode           -> OpenCV grayscale array
  2. Auto-Centering & Crop   -> Find bounding box with 15% padding
  3. High-Res Normalization  -> Create 512x512 master for Frontend display
  4. AI Tensor Preparation   -> Downscale 512 -> 224 for ViT Inference
  5. ImageNet normalization  -> (1, 3, 224, 224) float32 PyTorch tensor
"""

import base64
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def crop_and_center_image(img: np.ndarray, target_size: int = 512, padding_ratio: float = 0.15) -> np.ndarray:
    """
    Finds the drawing's bounding box and centers it within a square frame.
    Uses a larger target_size to ensure high quality on high-DPI displays.
    """
    # 1. Temporarily detect ink to find bounding box
    is_white_bg = np.mean(img) > 127
    temp_bin = cv2.bitwise_not(img) if is_white_bg else img
    
    # Simple threshold to find ink locations
    _, thresh = cv2.threshold(temp_bin, 10, 255, cv2.THRESH_BINARY)
    pts = cv2.findNonZero(thresh)
    
    if pts is None:
        return cv2.resize(img, (target_size, target_size))

    x, y, w, h = cv2.boundingRect(pts)
    
    # 2. Create a square container based on the largest dimension
    side = max(w, h)
    pad = int(side * padding_ratio)
    new_side = side + (2 * pad)
    
    # 3. Create clean white background (matching training data style)
    canvas = np.full((new_side, new_side), 255, dtype=np.uint8)
    
    # 4. Extract the drawing and ensure it is Black strokes on White background
    crop = img[y:y+h, x:x+w]
    if not is_white_bg:
        crop = cv2.bitwise_not(crop)
    
    # 5. Paste into center of the new canvas
    offset_x = pad + (side - w) // 2
    offset_y = pad + (side - h) // 2
    canvas[offset_y:offset_y+h, offset_x:offset_x+w] = crop
    
    # 6. Resize to the target display resolution (512 is sharp enough for web)
    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)

def prepare_image_for_vit(image_b64: str) -> tuple[torch.Tensor, Image.Image, str]:
    """
    Transforms base64 digital-ink into a dual-purpose resolution pipeline.
    Returns:
        - tensor: 224x224 for the AI model
        - pil_img: 224x224 for XAI heatmap generation
        - processed_b64: 512x512 high-quality image for the UI
    """
    # 1. Decode Base64 to Grayscale
    try:
        encoded_data = image_b64.split(',')[1] if ',' in image_b64 else image_b64
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    except Exception:
        raise ValueError("Failed to decode base64 image")

    if img_raw is None:
        raise ValueError("Decoded image is empty")

    # 2. Create High-Res Display version (512x512)
    # This solves the 'blurry' issue on the report screen.
    img_display = crop_and_center_image(img_raw, target_size=512)

    # 3. Create AI-ready version (224x224)
    img_vit = cv2.resize(img_display, (224, 224), interpolation=cv2.INTER_AREA)

    # 4. Encode the High-Res version to Base64 for the frontend
    _, buffer = cv2.imencode('.png', img_display)
    processed_b64 = f"data:image/png;base64,{base64.b64encode(buffer).decode()}"

    # 5. Prepare the 224x224 version for PyTorch ViT Model
    pil_img_vit = Image.fromarray(img_vit).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = preprocess(pil_img_vit).unsqueeze(0)

    return tensor, pil_img_vit, processed_b64