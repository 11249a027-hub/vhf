import cv2
import numpy as np

IMG_WIDTH = 220
IMG_HEIGHT = 155

def preprocess_signature(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)   # shape -> (1, H, W)

    return image