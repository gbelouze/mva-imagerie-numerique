import numpy as np
from scipy.ndimage import uniform_filter  # type: ignore


def guided_filter(im_P: np.ndarray, im_I: np.ndarray, *args, **kwargs) -> np.ndarray:
    assert 2 <= im_P.ndim <= 3
    assert 2 <= im_I.ndim <= 3

    if im_P.ndim == 3:
        return np.stack(
            [
                guided_filter(channel_P, im_I, *args, **kwargs)
                for channel_P in np.transpose(im_P, (2, 0, 1))
            ],
            axis=-1,
        )

    if im_I.ndim == 3:
        return guided_filter_rgb(im_P, im_I, *args, **kwargs)
    else:
        return guided_filter_gray(im_P, im_I, *args, **kwargs)


def guided_filter_gray(
    im_P: np.ndarray, im_I: np.ndarray, r: int, eps: float, average_in_window: bool = True
) -> np.ndarray:
    assert im_I.ndim == 2
    assert im_P.ndim == 2

    size = 2 * r + 1

    mean_I = uniform_filter(im_I, size)
    mean_P = uniform_filter(im_P, size)
    mean_IP = uniform_filter(im_I * im_P, size)
    var_I = uniform_filter(im_I ** 2, size) - mean_I ** 2

    a = (mean_IP - mean_I * mean_P) / (var_I + eps)
    b = mean_P - a * mean_I

    if average_in_window:
        a = uniform_filter(a, size)
        b = uniform_filter(b, size)
    return a * im_I + b


def guided_filter_rgb(
    im_P: np.ndarray, im_I: np.ndarray, r: int, eps: float, average_in_window: bool = True
) -> np.ndarray:
    assert im_I.ndim == 3
    assert im_P.ndim == 2

    size = 2 * r + 1

    mean_I = uniform_filter(im_I, (size, size, 0))
    mean_P = uniform_filter(im_P, size)
    mean_IP = uniform_filter(im_I * im_P[:, :, None], (size, size, 0))

    II = np.einsum("ijk,ijl->ijkl", im_I, im_I)
    meanImeanI = np.einsum("ijk,ijl->ijkl", mean_I, mean_I)

    sigma_I = uniform_filter(II, (size, size, 0, 0)) - meanImeanI

    a = np.einsum(
        "ijkl,ijl->ijk",
        np.linalg.inv(sigma_I + eps * np.identity(3)),
        mean_IP - mean_I * mean_P[:, :, None],
    )
    b = mean_P - np.einsum("ijk,ijk->ij", a, mean_I)

    if average_in_window:
        a = uniform_filter(a, (size, size, 0))
        b = uniform_filter(b, size)

    return np.einsum("ijk,ijk->ij", a, im_I) + b
