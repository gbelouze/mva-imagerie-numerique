from typing import Tuple, List, Optional

import numpy as np
from gf.filters import guided_filter
from scipy.ndimage import uniform_filter, laplace, gaussian_filter  # type: ignore


def decompose(
    im: np.ndarray, multichannel: Optional[bool] = None, k: int = 31
) -> Tuple[np.ndarray, np.ndarray]:
    if multichannel is None:
        multichannel = im.ndim == 3
    size = [k, k, 0] if multichannel else k
    b = uniform_filter(im, size=size)
    d = im - b
    return b, d


class gff:
    """Implementation of the image fusion with guided filtering method."""

    def __init__(self, ims: List[np.ndarray], color=None):
        if color is None:
            assert 2 <= ims[0].ndim <= 3
            color = "rgb" if ims[0].ndim == 3 else "gray"
        assert color in ["gray", "rgb", "hsv"]
        self.ims = ims
        self.color = color
        self.multichannel = color != "gray"

    def decompose(self, im: np.ndarray, k: int = 31) -> Tuple[np.ndarray, np.ndarray]:
        return decompose(im, self.multichannel, k)

    def saliency(self, im: np.ndarray, sigma: float = 5.0, r: int = 5) -> np.ndarray:
        if self.color == "rgb":
            im = im @ np.array([0.2989, 0.5870, 0.1140])  # rgb weights
        elif self.color == "hsv":
            im = im[:, :, 2]  # V channel
        h = laplace(im)
        s = gaussian_filter(np.abs(h), sigma, truncate=r / sigma)
        return s

    def weight_maps(self, ims: List[np.ndarray]) -> List[np.ndarray]:
        saliencies = np.stack([self.saliency(im) for im in ims], axis=0)
        weights = (saliencies == np.max(saliencies, axis=0)[None, :]).astype(np.float64)
        return [w for w in weights]

    def fusion(self, r1=45, r2=7, eps1=0.3, eps2=1e-6, filt=guided_filter):
        bs, ds = zip(*[self.decompose(im) for im in self.ims])
        weights = self.weight_maps(self.ims)

        weights_b = np.stack([filt(p, i, r1, eps1) for p, i in zip(weights, self.ims)])
        weights_d = np.stack([filt(p, i, r2, eps2) for p, i in zip(weights, self.ims)])
        weights_b = weights_b / np.sum(weights_b, axis=0)
        weights_d = weights_d / np.sum(weights_b, axis=0)
        b_bar = sum(w[:, :, None] * b for w, b in zip(weights_b, bs))
        d_bar = sum(w[:, :, None] * d for w, d in zip(weights_d, ds))
        out = b_bar + d_bar
        if self.color == "rgb" and (np.max(out) > 1 or np.min(out) < 0):
            out = (out - np.min(out)) / (np.max(out) - np.min(out))
        return out

    def fusion_without_separation(self, r=45, eps=0.3):
        """Fusion without base/detail separation"""
        weights = np.stack(
            [guided_filter(p, i, r, eps) for p, i in zip(self.weight_maps(self.ims), self.ims)]
        )
        weights = weights / np.sum(weights, axis=0)
        out = sum(w[:, :, None] * b for w, b in zip(weights, self.ims))
        if self.color == "rgb" and (np.max(out) > 1 or np.min(out) < 0):
            out = (out - np.min(out)) / (np.max(out) - np.min(out))
        return out
