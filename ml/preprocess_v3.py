import os
import cv2
import numpy as np
import torch

TARGET_WIDTH = 220
TARGET_HEIGHT = 155

def preprocess_signature_v3(image_path: str, target_size=(TARGET_WIDTH, TARGET_HEIGHT)) -> np.ndarray:
    """
    Enhanced signature preprocessing:
    1. Grayscale reading
    2. Otsu thresholding for precise stroke bounding-box detection (strips whitespace margins)
    3. Inverted grayscale normalization: background is 0.0, ink strokes retain pressure anti-aliasing
    4. Aspect-ratio preserving letterbox resize (prevents distortion of signature slant and aspect ratio)
    5. Returns shape (1, H, W) as float32 in [0, 1]
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Detect ink bounding box using Otsu
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = np.where(binary > 0)
    
    if len(pts[0]) > 0:
        pad = 6
        ymin = max(0, pts[0].min() - pad)
        ymax = min(img.shape[0], pts[0].max() + pad)
        xmin = max(0, pts[1].min() - pad)
        xmax = min(img.shape[1], pts[1].max() + pad)
        cropped_gray = img[ymin:ymax, xmin:xmax]
    else:
        cropped_gray = img

    # Inverted grayscale: 0.0 for white paper background, >0 for dark ink
    inverted = (255.0 - cropped_gray.astype(np.float32)) / 255.0

    # Aspect-ratio preserved letterbox resize
    target_w, target_h = target_size
    h, w = inverted.shape
    scale = min(target_w / max(1, w), target_h / max(1, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(inverted, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    # Shape: (1, H, W)
    return np.expand_dims(canvas, axis=0)

def build_signature_cache(full_org_path: str, full_forg_path: str) -> dict:
    """Preprocess and cache all CEDAR images in memory (~350 MB RAM)."""
    cache = {}
    org_files = [os.path.join(full_org_path, f) for f in os.listdir(full_org_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    forg_files = [os.path.join(full_forg_path, f) for f in os.listdir(full_forg_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_files = org_files + forg_files
    
    for path in all_files:
        norm_path = os.path.normpath(path).replace("\\", "/")
        cache[norm_path] = preprocess_signature_v3(path)
    return cache
