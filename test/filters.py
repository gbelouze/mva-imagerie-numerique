import numpy as np
from numba import njit
from scipy.ndimage import convolve


def mean_kernel(size: int, d: int) -> np.ndarray:
    shape = [size, size] + [1] * (d - 2)
    return np.ones((size, size))[None, :].reshape(shape) / size ** 2


def sum_kernel(size: int, d: int) -> np.ndarray:
    shape = [size, size] + [1] * (d - 2)
    return np.ones((size, size))[None, :].reshape(shape)


def guided_filter(im_P: np.ndarray, im_I: np.ndarray, r: int, eps: float) -> np.ndarray:
    if len(im_P.shape) > 2:
        return np.stack([guided_filter(ch, im_I, r, eps)
                         for ch in np.moveaxis(im_P, -1, 0)], axis=-1)
    size = 2 * r + 1
    k_mean = mean_kernel(size, 2)

    mean_I = convolve(im_I, k_mean)
    mean_P = convolve(im_P, k_mean)
    mean_IP = convolve(im_I * im_P, k_mean)
    var_I = convolve(im_I ** 2, k_mean) - mean_I ** 2

    a = (mean_IP - mean_I * mean_P) / (var_I + eps)
    b = mean_P - a * mean_I

    a = convolve(a, k_mean)
    b = convolve(b, k_mean)
    return a * im_I + b


@njit
def fast_sig(im_I: np.ndarray, r: int) -> np.ndarray:
    m, n, _ = im_I.shape
    sig = np.zeros((m, n, 3, 3))
    for i in range(r, m - r - 1):
        for j in range(r, n - r - 1):
            sig[i, j, :, :] = np.cov(im_I[i - r: i + r + 1, j - r: j + r + 1, :].copy().reshape(3, -1))
    return sig


def guided_filter_rgb(im_P: np.ndarray, im_I: np.ndarray, r: int, eps: float) -> np.ndarray:
    rgb_I, rgb_P = len(im_I.shape) > 2, len(im_P.shape) > 2
    if rgb_P:
        return np.stack([guided_filter_rgb(ch, im_I, r, eps)
                         for ch in np.moveaxis(im_P, -1, 0)], axis=-1)
    m, n = im_P.shape
    size = 2 * r + 1

    k_mean = mean_kernel(size, 2)
    k_mean_3d = mean_kernel(size, 3)

    mean_I = convolve(im_I, k_mean_3d)
    mean_P = convolve(im_P, k_mean)
    mean_IP = convolve(im_I * im_P[None, :].reshape(m, n, 1), k_mean_3d)

    """
    mean_IIT = convolve(np.einsum('ijk,ijl->ijkl', im_I, im_I),
                        mean_kernel(size, 4))
    sig = (mean_IIT -
           np.einsum('ijk,ijl->ijkl', mean_I, mean_I))  # NOT SURE
    """

    sig = fast_sig(im_I, r)  # TODO : vectorize

    a = np.einsum('ijkl,ijl->ijk',
                  sig + eps * np.eye(3),
                  (mean_IP - mean_I * mean_P[None, :].reshape(m, n, 1)))

    b = mean_P - np.einsum('ijk,ijk->ij', a, mean_I)

    a = convolve(a, k_mean_3d)
    b = convolve(b, k_mean)

    return np.einsum('ijk,ijk->ij', a, im_I) + b
