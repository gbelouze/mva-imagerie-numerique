from gf import filters
import numpy as np
from imageio import imread
import os


def test_anisotropy_rgb_guide():
    """filters should be covariant with pi/2 rotations"""
    im = imread(f"{os.path.dirname(__file__)}/../data/lytro/1/lytro-01-A.jpg") / 255
    im_t = np.transpose(im, (1, 0, 2))
    out = filters.guided_filter(im, im, 10, 0.1)
    out_t = filters.guided_filter(im_t, im_t, 10, 0.1)
    diff = np.abs(out - np.transpose(out_t, (1, 0, 2))).max()
    assert diff < 1e-6


def test_anisotropy_gray_guide():
    """filters should be covariant with pi/2 rotations"""
    im = imread(f"{os.path.dirname(__file__)}/../data/petrovic/fused001_1.tif") / 255
    im_t = np.transpose(im, (1, 0))
    out = filters.guided_filter(im, im, 10, 0.1)
    out_t = filters.guided_filter(im_t, im_t, 10, 0.1)
    diff = np.abs(out - np.transpose(out_t, (1, 0))).max()
    assert diff < 1e-6
