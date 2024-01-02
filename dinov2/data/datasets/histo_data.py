# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, Any
import openslide
from tqdm import tqdm
import pandas as pd
from PIL import Image
import h5py

import torch
from torchvision.datasets import VisionDataset
from torchvision import transforms

import numpy as np

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
        # root: str = "/lustre/groups/shared/histology_data/TCGA/CRC/patches/512px_crc_wonorm_complete_diag_frozen.txt",
        root: str = "/lustre/groups/shared/histology_data/TCGA",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        if Path(root).is_file():
            self.patches = np.loadtxt(root, dtype=str)
        else:
            # self.patches = list(Path(root).glob("**/*.jpeg"))
            self.patches = list(Path(root).glob("**/patch*/0512_px_20_mag*/*DX*.h5"))
            np.savetxt(f"{root}_h5patches.txt", self.patches, delimiter="\n", fmt='%s')

        self.patches_files = [h5py.File(patch, 'r') for patch in self.patches]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index, e, self.patches[index]}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    def get_image_data(self, index: int) -> Image:

        # load image from h5 file
        # file = h5py.File(self.patches[index], 'r')
        file = self.patches_files[index]
        patch_id = torch.randint(file['data'].shape[0], (1,)).item()
        patch = Image.fromarray(file['data'][patch_id])

        # load image from jpeg file
        # patch = Image.open(self.patches[index]).convert(mode="RGB")
        
        # random crop to (256, 256)
        # h, w = patch.size
        # i = torch.randint(0, h - 256 + 1, size=(1,)).item()
        # j = torch.randint(0, w - 256 + 1, size=(1,)).item()
        # patch = transforms.functional.crop(patch, i, j, 256, 256)

        return patch

    def get_target(self, index: int) -> torch.Tensor:
        # labels are not used for training
        return torch.zeros((1,))

    def __len__(self) -> int:
        # assert len(entries) == self.split.length
        return len(self.patches)