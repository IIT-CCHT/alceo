from typing import Any, Dict, List
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from pathlib import Path
from glob import glob
from torchvision.io import read_image


import torch
import torchvision.transforms.functional as tvf


def encode_color(mask: torch.Tensor):
    assert torch.iinfo(mask.dtype).bits >= 32
    return mask[0] + (256 * mask[1]) + ((256 * 256) * mask[2])


@dataclass
class SECONDataset(Dataset):
    """A PyTorch Dataset for loading the [SEmantic Change detectiON Dataset](http://www.captain-whu.com/project/SCD/). 
    
    Args:
        dataset_root (Path): The path to the root folder containing SECOND.
        is_train (bool): If it should load the training or testing split. Defaults to True.
    """
    
    dataset_root: Path
    is_train: bool = True
    image_ids: List[str] = field(init=False)

    label_colors = torch.tensor(
        [
            [0, 0, 0, 128, 128, 255, 255],
            [0, 128, 255, 0, 128, 0, 255],
            [255, 0, 0, 0, 128, 0, 255],
        ],
        dtype=torch.int,
    )

    label_repr = encode_color(label_colors).view(7, 1, 1)

    label_names = [
        "water",
        "low_vegetation",
        "tree",
        "building",
        "no_vegetation_surface",
        "playground",
        "no_change",
    ]

    label_colors = [
        "blue",
        "darkgreen",
        "green",
        "brown",
        "gray",
        "red",
        "black",
    ]

    def __post_init__(self):
        if self.is_train:
            self.dataset_root = Path(self.dataset_root) / "train"
        else:
            self.dataset_root = Path(self.dataset_root) / "test"

        im1_found = glob(str(self.dataset_root / "im1" / "*.png"))
        size = 40
        self.image_ids = [Path(im1).stem for im1 in im1_found]
        self.image_ids.sort()
        self.image_ids = self.image_ids[0:size]

    def __len__(self):
        return len(self.image_ids)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        _image = read_image(str(image_path)).to(dtype=torch.float)
        _image = tvf.normalize(
            _image, mean=[101.0625, 105.7892, 103.1601], std=[41.7312, 40.8010, 40.7497]
        )
        return _image

    def _load_mask(self, mask_path) -> torch.Tensor:
        _mask = read_image(str(mask_path))
        _multiclass_mask = (
            (encode_color(_mask.int()) == self.label_repr).int().argmax(dim=0)
        )
        return _multiclass_mask

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Loads a single change detection tile associated to the `index`

        Args:
            index (int): index of the tile to load.

        Returns:
            Dict[str, Any]: A dictionary with keys: "image_id" (tile filename without extension), "im1" and "im2" (Tensors of the images at time T1 and T2 respectively), "label1" and "label2" (semantic change detection ground truth as per SECOND format). 
        """
        image_id = self.image_ids[index]
        im1_path = self.dataset_root / "im1" / f"{image_id}.png"
        im2_path = self.dataset_root / "im2" / f"{image_id}.png"

        label1_path = self.dataset_root / "label1" / f"{image_id}.png"
        label2_path = self.dataset_root / "label2" / f"{image_id}.png"

        im1 = self._load_image(im1_path)
        im2 = self._load_image(im2_path)

        label1 = self._load_mask(label1_path)
        label2 = self._load_mask(label2_path)

        return {
            "im1": im1,
            "im2": im2,
            "label1": label1,
            "label2": label2,
            "image_id": image_id,
        }


__all__ = [SECONDataset.__name__]
