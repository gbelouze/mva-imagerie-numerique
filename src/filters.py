import numpy as np
from scipy.ndimage import uniform_filter


def guided_filter(im_P: np.ndarray, im_I: np.ndarray, r: int, eps: float) -> np.ndarray:
    if len(im_I.shape) > 2:
        return guided_filter_rgb(im_P, im_I, r, eps)
    else:
        return guided_filter_gray(im_P, im_I, r, eps)


def guided_filter_gray(im_P: np.ndarray, im_I: np.ndarray, r: int, eps: float) -> np.ndarray:
    if len(im_P.shape) > 2:
        return np.stack([guided_filter(ch, im_I, r, eps)
                         for ch in np.moveaxis(im_P, -1, 0)], axis=-1)

    size = 2 * r + 1, 2

    mean_I = uniform_filter(im_I, size)
    mean_P = uniform_filter(im_P, size)
    mean_IP = uniform_filter(im_I * im_P, size)
    var_I = uniform_filter(im_I ** 2, size) - mean_I ** 2

    a = (mean_IP - mean_I * mean_P) / (var_I + eps)
    b = mean_P - a * mean_I

    a = uniform_filter(a, size)
    b = uniform_filter(b, size)
    return a * im_I + b


def guided_filter_rgb(im_P: np.ndarray, im_I: np.ndarray, r: int, eps: float) -> np.ndarray:
    if len(im_P.shape) > 2:
        return np.stack([guided_filter_rgb(ch, im_I, r, eps)
                         for ch in np.moveaxis(im_P, -1, 0)], axis=-1)
    m, n = im_P.shape
    size = 2 * r + 1

    mean_I = uniform_filter(im_I, (size, size, 0))
    mean_P = uniform_filter(im_P, size)
    mean_IP = uniform_filter(im_I * im_P[None, :].reshape(m, n, 1), (size, size, 0))

    sig = uniform_filter(np.einsum('ijk,ijl->ijkl', im_I, im_I),
                         (size, size, 0, 0)) * (size ** 2 / (size ** 2 - 1))

    a = np.einsum('ijkl,ijl->ijk',
                  sig + eps * np.identity(3),
                  (mean_IP - mean_I * mean_P[None, :].reshape(m, n, 1)))

    b = mean_P - np.einsum('ijk,ijk->ij', a, mean_I)

    a = uniform_filter(a, (size, size, 0))
    b = uniform_filter(b, size)

    return np.einsum('ijk,ijk->ij', a, im_I) + b
