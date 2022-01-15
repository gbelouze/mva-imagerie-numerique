from typing import List

import numpy as np
from gf.filters import guided_filter
from gf.fusion import weight_maps


def fusion_without_separation(ims: List[np.ndarray], r=45, eps=0.3):
    weights = np.stack([guided_filter(p, i, r, eps) for p, i in zip(weight_maps(ims), ims)])
    weights = weights / np.sum(weights, axis=0)
    out = sum(w[:, :, None] * b for w, b in zip(weights, ims))
    if np.max(out) > 1 or np.min(out) < 0:
        out = (out - np.min(out)) / (np.max(out) - np.min(out))
    return out
