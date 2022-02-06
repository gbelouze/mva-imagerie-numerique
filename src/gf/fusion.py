from typing import Tuple, List, Optional

import numpy as np
from gf.filters import guided_filter
from scipy.ndimage import uniform_filter, laplace, gaussian_filter  # type: ignore


def decompose(
    im: np.ndarray, multichannel: Optional[bool] = None, k: int = 31
) -> Tuple[np.ndarray, np.ndarray]:
    """base / detail layer decomposition"""
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

        self.compute_weight_maps()
        self.refined_weights = {"base": np.empty(0), "detail": np.empty(0), "full": np.empty(0)}
        self.normalised_refined_weights = {"base": np.empty(0),
            "detail": np.empty(0), "full": np.empty(0)}
        self.fused = {"base": np.empty(0), "detail": np.empty(0), "full": np.empty(0)}

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

    def compute_weight_maps(self):
        self.saliencies = np.stack([self.saliency(im) for im in self.ims], axis=0)
        self.weights = (self.saliencies == np.max(self.saliencies, axis=0)[None, :]).astype(
            np.float64
        )

    def fusion(self, r1=45, r2=7, eps1=0.3, eps2=1e-6, filt=guided_filter):
        bs, ds = zip(*[self.decompose(im) for im in self.ims])
        weights = [w for w in self.weights]

        self.refined_weights["base"] = weights_b = np.stack(
            [filt(p, i, r1, eps1) for p, i in zip(weights, self.ims)]
        ).clip(0, 1)
        self.refined_weights["detail"] = weights_d = np.stack(
            [filt(p, i, r2, eps2) for p, i in zip(weights, self.ims)]
        ).clip(0, 1)
        self.normalised_refined_weights["base"] = weights_b = weights_b / np.sum(weights_b, axis=0)
        self.normalised_refined_weights["detail"] = weights_d = weights_d / np.sum(
            weights_b, axis=0
        )
        if self.multichannel:
            self.fused["base"] = b_bar = sum(w[:, :, None] * b for w, b in zip(weights_b, bs))
            self.fused["detail"] = d_bar = sum(w[:, :, None] * d for w, d in zip(weights_d, ds))
        else:
            self.fused["base"] = b_bar = sum(w * b for w, b in zip(weights_b, bs))
            self.fused["detail"] = d_bar = sum(w * d for w, d in zip(weights_d, ds))
        out = np.clip(b_bar + d_bar, 0, 1)
        return out

    def fusion_without_separation(self, r=45, eps=0.3, filt=guided_filter):
        """Fusion without base/detail separation"""
        self.refined_weights["full"] = weights = np.stack(
            [filt(p, i, r, eps) for p, i in zip(self.weights, self.ims)]
        ).clip(0, 1)
        self.normalised_refined_weights["full"] = weights = weights / np.sum(weights, axis=0)
        if self.multichannel:
            self.fused["full"] = out = sum(w[:, :, None] * i for w, i in zip(weights, self.ims))
        else:
            self.fused["full"] = out = sum(w * i for w, i in zip(weights, self.ims))
        out = np.clip(out, 0, 1)
        return out
