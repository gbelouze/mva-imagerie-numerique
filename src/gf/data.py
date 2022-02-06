import numpy as np
import imageio as iio  # type: ignore
from pathlib import Path

from typing import List, Union, Tuple


# type imread function of imageio
def _imread(path: Path) -> np.ndarray:
    return iio.imread(path.as_posix()) / 255


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix in [".tif", ".jpeg", ".jpg", ".png"]


def _group_images_in_dir(path: Path) -> Tuple[str, List[Path]]:
    return path.name, [file for file in path.iterdir() if _is_image(file)]


def _all_dirs(path: Path) -> List[Path]:
    """Finds all subdirectories of path"""
    assert path.is_dir()
    rep = []
    for subpath in path.iterdir():
        if subpath.is_dir():
            rep.append(subpath)
            rep += _all_dirs(subpath)
    return rep


class MultiviewDataset:
    """Datasets with several views per image."""

    def __init__(self, path: Path):
        """explores directory [path] and tries to group images based on file names"""
        self.path = path
        self.groups: dict[str, List[np.ndarray]] = {}
        for dir in _all_dirs(path):
            group_name, images = _group_images_in_dir(dir)
            assert len(images) > 0, f"Directory {dir} does not hold images"
            assert group_name not in self.groups, "Directory names should be unique"
            self.groups[group_name] = [_imread(image) for image in images]
        self.keys = list(self.groups.keys())

        assert self.well_formed()

    def well_formed(self) -> bool:
        """assert images have all identical sizes within each group"""
        for images in self.groups.values():
            shape = images[0].shape
            if not all(image.shape == shape for image in images):
                return False
        return True

    def __iter__(self):
        yield from self.groups.items()

    def __getitem__(self, index: Union[int, str]):
        if isinstance(index, int):
            return self.groups[self.keys[index]]
        return self.groups[index]

    def __len__(self):
        return len(self.groups)

    def __repr__(self):
        return str(self.keys)
