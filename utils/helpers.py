from skimage.morphology import remove_small_holes, remove_small_objects, binary_closing
import numpy as np


def morpholoy_post_process(patch: np.ndarray) -> np.ndarray:
    """Performs a sequence of morphogical operations on input image"""
    patch = remove_small_holes(patch)
    patch = remove_small_objects(patch)
    patch = binary_closing(patch)
    return patch


def imagenet_normalise(img: np.ndarray) -> np.ndarray:
    """Normalises input image to ImageNet mean and std"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img - mean
    img = img / std
    return img
