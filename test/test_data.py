from gf import data
from pathlib import Path

dataset1 = Path("data/ours")
dataset2 = Path("data/lytro")
dataset3 = Path("data/MEFDatabase/source/")


def test_load_dataset():
    _ = data.MultiviewDataset(dataset1)
    _ = data.MultiviewDataset(dataset2)
    _ = data.MultiviewDataset(dataset3)
