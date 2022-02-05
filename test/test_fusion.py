from gf import fusion
import numpy as np
from imageio import imread
import os

im_gray = imread(f"{os.path.dirname(__file__)}/../data/petrovic/fused001_1.tif") / 255
im_rgb = imread(f"{os.path.dirname(__file__)}/../data/lytro/lytro-01-A.jpg") / 255


def test_decompose_gray():
    b, d = fusion.decompose(im_gray)
    diff = np.abs(im_gray - (b + d)).max()
    assert diff < 1e-10


def test_decompose_rgb():
    b, d = fusion.decompose(im_rgb)
    diff = np.abs(im_rgb - (b + d)).max()
    assert diff < 1e-10


def test_fusion_black_black():
    black = np.zeros((100, 100))
    gff = fusion.gff([black, black])
    fused = gff.fusion()
    diff = np.abs(fused - black).max()
    assert diff < 1e-10


def test_fusion_black_white():
    black = np.zeros((100, 100))
    white = np.ones((100, 100))
    gff = fusion.gff([black, white])
    gff.fusion()
