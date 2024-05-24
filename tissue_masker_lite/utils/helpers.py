import cv2
import numpy as np


def morpholoy_post_process(patch: np.ndarray) -> np.ndarray:
    """Performs a sequence of morphogical operations on input image"""
    kernel_diameter = 72
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_diameter, kernel_diameter)
    )
    closing = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening


def imagenet_normalise(img: np.ndarray) -> np.ndarray:
    """Normalises input image to ImageNet mean and std"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img - mean
    img = img / std
    return img
