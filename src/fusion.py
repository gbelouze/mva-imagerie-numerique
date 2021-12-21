from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter, laplace, gaussian_filter


def decompose(im: np.ndarray, k: int = 31) -> Tuple[np.ndarray, np.ndarray]:
    shape = [k, k, 0] if len(im.shape) > 2 else k
    b = uniform_filter(im, shape)
    d = im - b
    return b, d


def saliency(im: np.ndarray, sig: float = 5) -> np.ndarray:
    if len(im.shape) > 2:
        return np.stack([saliency(ch, sig)
                         for ch in np.moveaxis(im, -1, 0)], axis=-1)
    h = laplace(im)
    s = gaussian_filter(np.abs(h), sig)
    return s
