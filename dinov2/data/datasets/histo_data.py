# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from .extended import ExtendedVisionDataset
from .image_net import _Split

logger = logging.getLogger("dinov2")


class WSIDataset(VisionDataset):

    def __init__(
        self,
        *,
        root: str = "/lustre/groups/shared/tcga/CRC/slides",
        load: bool = False,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.load = load

        if self.load:
            self.slides = []
            for slide in tqdm(Path(root).glob("*.svs")):
                slide = openslide.open_slide(str(slide))
                self.slides.append(slide)
        else:
            self.slides = list(Path(root).glob("*.svs"))
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    def get_image_data(self, index: int) -> Image:
        if self.load:
            slide = self.slides[index]
        else: 
            slide = openslide.open_slide(str(self.slides[index]))

        dim_x, dim_y = slide.dimensions
        center = (dim_x // 2, dim_y // 2)
        patch_size = 224

        # sample patch from image center
        patch = slide.read_region((center[0] - patch_size // 2, center[1] - patch_size // 2), 0, (patch_size, patch_size)).convert(mode="RGB")

        return patch

    def get_target(self, index: int) -> torch.Tensor:
        # labels are not used for training
        return torch.zeros((1,))

    def __len__(self) -> int:
        # assert len(entries) == self.split.length
        return len(self.slides)


class PatchDataset(VisionDataset):

    def __init__(
        self,
        *,
        root: str = "/lustre/groups/shared/histology_data/TCGA/CRC/patches/512px_crc_wonorm_complete_diag_frozen.txt",
        # root: str = "/lustre/groups/shared/histology_data/TCGA",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        if Path(root).is_file():
            self.patches = np.loadtxt(root, dtype=str)
        else:
            self.patches = list(Path(root).glob("**/*.jpeg"))
            np.savetxt(f"{root}_jpeg_patches.txt", self.patches, delimiter="\n", fmt='%s')

    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            print(f"can not read image for sample {index, e, self.patches[index]}")
            return self.__getitem__(index+1)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    def get_image_data(self, index: int) -> Image:

        # load image from jpeg file
        patch = Image.open(self.patches[index]).convert(mode="RGB")
        
        # random crop to (256, 256)
        # h, w = patch.size
        # i = torch.randint(0, h - 224 + 1, size=(1,)).item()
        # j = torch.randint(0, w - 224 + 1, size=(1,)).item()
        # patch = transforms.functional.crop(patch, i, j, 224, 224)

        # random crop to any size between original size and (224, 224) > resize to (224, 224)
        h, w = patch.size
        size = torch.randint(224, max(h, w) + 1, size=(1,)).item()
        i = torch.randint(0, h - size + 1, size=(1,)).item()
        j = torch.randint(0, w - size + 1, size=(1,)).item()
        patch = transforms.functional.crop(patch, i, j, size, size)
        patch = transforms.functional.resize(patch, (224, 224))

        return patch

    def get_target(self, index: int) -> torch.Tensor:
        # labels are not used for training
        return torch.zeros((1,))

    def __len__(self) -> int:
        # assert len(entries) == self.split.length
        return len(self.patches)
