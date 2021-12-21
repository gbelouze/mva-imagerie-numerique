from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter, laplace, gaussian_filter  # type: ignore

from filters import guided_filter


def decompose(im: np.ndarray, k: int = 31) -> Tuple[np.ndarray, np.ndarray]:
    shape = [k, k, 0] if len(im.shape) > 2 else k
    b = uniform_filter(im, shape)
    d = im - b
    return b, d


def saliency(im: np.ndarray, sig: float = 5) -> np.ndarray:
    if len(im.shape) > 2:
        im = im @ np.array([0.2989, 0.5870, 0.1140])  # rgb weights
    h = laplace(im)
    s = gaussian_filter(np.abs(h), sig)
    return s


def weight_maps(ims: list) -> list:
    saliencies = np.stack([saliency(im) for im in ims], axis=0)
    p, m, n = saliencies.shape
    argmax_saliencies = np.argmax(saliencies, axis=0)
    weights = np.zeros((p, m, n))
    for (i, j), k in np.ndenumerate(argmax_saliencies):
        weights[k, i, j] = 1
    return [w for w in weights]


def fusion(ims: list, r1=45, r2=7, eps1=0.3, eps2=1e-6):
    bs, ds = zip(*[decompose(im) for im in ims])
    weights = weight_maps(ims)
    weights_b = [guided_filter(p, i, r1, eps1) for p, i in zip(weights, ims)]
    weights_d = [guided_filter(p, i, r2, eps2) for p, i in zip(weights, ims)]
    b_bar = sum(w[:, :, None] * b for w, b in zip(weights_b, bs))
    d_bar = sum(w[:, :, None] * d for w, d in zip(weights_d, ds))
    return b_bar + d_bar
