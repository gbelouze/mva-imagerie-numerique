from typing import Tuple, List

import numpy as np
from gf.filters import guided_filter
from scipy.ndimage import uniform_filter, laplace, gaussian_filter  # type: ignore


def rgb(im: np.ndarray) -> bool:
    assert 2 <= len(im.shape) <= 3
    return len(im.shape) == 3


def decompose(im: np.ndarray, k: int = 31) -> Tuple[np.ndarray, np.ndarray]:
    size = [k, k, 0] if rgb(im) else k
    b = uniform_filter(im, size=size)
    d = im - b
    return b, d


def saliency(im: np.ndarray, sigma: float = 5.0, r: int = 5) -> np.ndarray:
    if rgb(im):
        im = im @ np.array([0.2989, 0.5870, 0.1140])  # rgb weights
    h = laplace(im)
    s = gaussian_filter(np.abs(h), sigma, truncate=r / sigma)
    return s


def weight_maps(ims: List[np.ndarray]) -> List[np.ndarray]:
    saliencies = np.stack([saliency(im) for im in ims], axis=0)
    weights = (saliencies == saliencies.max(axis=0)[None, :]).astype(np.float64)
    return [w for w in weights]


def fusion(ims: List[np.ndarray], r1=45, r2=7, eps1=0.3, eps2=1e-6):
    bs, ds = zip(*[decompose(im) for im in ims])
    weights = weight_maps(ims)

    weights_b = np.stack([guided_filter(p, i, r1, eps1) for p, i in zip(weights, ims)])
    weights_d = np.stack([guided_filter(p, i, r2, eps2) for p, i in zip(weights, ims)])
    weights_b = weights_b / np.sum(weights_b, axis=0)
    weights_d = weights_d / np.sum(weights_b, axis=0)
    b_bar = sum(w[:, :, None] * b for w, b in zip(weights_b, bs))
    d_bar = sum(w[:, :, None] * d for w, d in zip(weights_d, ds))
    out = b_bar + d_bar
    return (out - np.min(out)) / (np.max(out) - np.min(out))
