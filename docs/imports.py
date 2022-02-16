import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import cv2
from scipy.ndimage import gaussian_filter as gaussian
from pathlib import Path

# Our own implementation
import gf
import gf.filters as filters
import gf.data as data
import gf.fusion as fusion
import gf.recalage as recalage

import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 160
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.titlesize"] = "small"


def gray_to_rgb(im):
    return np.stack([im, im, im], axis=-1)


def plot_images(*ims, maxwidth=5, labels=None, title=""):
    import logging

    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)  # remove clipping input warning

    max_ndim = max([im.ndim for im in ims])
    if max_ndim == 3:
        ims = [im if im.ndim == 3 else gray_to_rgb(im) for im in ims]
    else:
        ims = [im for im in ims]

    n = len(ims)
    nrows, ncols = (n - 1) // maxwidth + 1, min(maxwidth, n)
    h, w, *_ = ims[0].shape
    w, h = (10 * w) / (w + h), (10 * h) / (w + h)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize=(ncols * w, nrows * h))
    if labels is None:
        labels = [None] * n
    for ind, (label, im) in enumerate(zip(labels, ims)):
        ax = axs[ind // ncols, ind % ncols]
        ax.imshow(im, cmap=plt.gray())
        if label is not None:
            ax.set_title(label)
        ax.set_axis_off()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(title)
    logger.setLevel(old_level)


BASE_DIR = ".."

multi_exposure_dataset = data.MultiviewDataset(Path(f"{BASE_DIR}/data/MEFDatabase/source/"))
multi_focus_dataset = data.MultiviewDataset(Path(f"{BASE_DIR}/data/lytro"))
ours_dataset = data.MultiviewDataset(Path(f"{BASE_DIR}/data/ours"))

multi_exposure_sample = multi_exposure_dataset["Balloons_Erik Reinhard"]
multi_exposure_sample = multi_exposure_dataset["Lighthouse_HDRsoft"]
multi_focus_sample = multi_focus_dataset["20"]
ours_sample = ours_dataset["stylos2"]
gray_sample = [
    iio.imread(path) / 255
    for path in [f"{BASE_DIR}/data/petrovic/input001_{i}.tif" for i in (1, 2)]
]
