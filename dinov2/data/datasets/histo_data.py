# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import itertools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import openslide
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset
from tqdm import tqdm

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
        patch = slide.read_region(
            (center[0] - patch_size // 2, center[1] - patch_size // 2), 0, (patch_size, patch_size)
        ).convert(mode="RGB")

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
            print("Loading ", root)
            self.patches = np.loadtxt(root, dtype=str)
        else:
            self.patches = list(Path(root).glob("**/*.jpeg"))
            np.savetxt(f"{root}_jpeg_patches.txt", self.patches, delimiter="\n", fmt="%s")
        np.random.shuffle(self.patches)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            print(f"can not read image for sample {index, e, self.patches[index]}")
            return self.__getitem__(index + 1)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_image_data(self, index: int, min_dimension=224) -> Image:

        # load image from file
        patch = Image.open(self.patches[index]).convert(mode="RGB")

        h, w = patch.size

        # random crop to (300, 300)
        crop_size = 300
        if h > crop_size and w > crop_size:
            i = torch.randint(0, h - crop_size + 1, size=(1,)).item()
            j = torch.randint(0, w - crop_size + 1, size=(1,)).item()
            patch = transforms.functional.crop(patch, i, j, crop_size, crop_size)

        if h < min_dimension or w < min_dimension:

            print("Image had to be resized due to smaller size than 224: ", self.patches[index])

            if w < h:
                new_width = min_dimension
                new_height = int((min_dimension / w) * h)

            else:
                new_height = min_dimension
                new_width = int((min_dimension / h) * w)

            patch = patch.resize((new_width, new_height), Image.Resampling.NEAREST)

        # random crop to any size between original size and (224, 224) > resize to (224, 224)
        # size = torch.randint(224, max(h, w) + 1, size=(1,)).item()
        # i = torch.randint(0, h - size + 1, size=(1,)).item()
        # j = torch.randint(0, w - size + 1, size=(1,)).item()
        # patch = transforms.functional.crop(patch, i, j, size, size)
        # patch = transforms.functional.resize(patch, (224, 224))

        return patch

    def __len__(self) -> int:
        return len(self.patches)

    def get_target(self, index: int) -> torch.Tensor:
        # labels are not used for training
        return torch.zeros((1,))


def arrange_files(file_paths):
    # Group files by their parent folder
    grouped_files = defaultdict(list)
    for file_path in file_paths:
        parent_folder = Path(file_path).parent.name
        grouped_files[parent_folder].append(file_path)

    # Create a balanced ordering of files
    balanced_ordering = []
    # Use itertools.zip_longest for round-robin style iteration
    for group in itertools.zip_longest(*grouped_files.values()):
        # Filter out 'None' in case some groups are smaller than others
        balanced_ordering.extend(filter(None, group))

    return balanced_ordering


class BalancedPatchDataset(VisionDataset):
    def __init__(
        self,
        *,
        root: str = "",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.patches = []
        self.dataset_sizes = []

        all_dataset_files = Path(root).glob("*.txt")

        for dataset_file in all_dataset_files:
            print("Loading ", dataset_file)
            with open(dataset_file, 'r') as file:
                content = file.read()
            file_list_unsorted = content.splitlines()

        file_list = arrange_files(file_list_unsorted)
        self.patches.append(file_list)
        self.dataset_sizes.append(int(len(file_list)))

        self.num_datasets = len(self.patches)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        dataset_index = index % self.num_datasets
        index_in_dataset = int(index / self.num_datasets) % self.dataset_sizes[dataset_index]

        try:

            image = self.get_image_data(dataset_index, index_in_dataset)

        except Exception as e:
            print(f"can not read image for sample {index, e,self.patches[dataset_index][index_in_dataset]}")
            return self.__getitem__(index + 1)

        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_image_data(self, dataset_index: int, index_in_dataset: int, min_dimension=224) -> Image:

        # load image from jpeg file
        filepath = self.patches[dataset_index][index_in_dataset]
        patch = Image.open(filepath).convert(mode="RGB")

        h, w = patch.size

        # random crop to (300, 300)
        crop_size = 300
        if h > crop_size and w > crop_size:
            i = torch.randint(0, h - crop_size + 1, size=(1,)).item()
            j = torch.randint(0, w - crop_size + 1, size=(1,)).item()
            patch = transforms.functional.crop(patch, i, j, crop_size, crop_size)

        if h < min_dimension or w < min_dimension:

            print("Image had to be resized due to smaller size than 224: ", filepath)

            if w < h:
                new_width = min_dimension
                new_height = int((min_dimension / w) * h)

            else:
                new_height = min_dimension
                new_width = int((min_dimension / h) * w)

            patch = patch.resize((new_width, new_height), Image.Resampling.NEAREST)

        return patch

    def get_target(self, index: int) -> torch.Tensor:
        # labels are not used for training
        return torch.zeros((1,))

    def __len__(self) -> int:
        # assert len(entries) == self.split.length
        return int(np.sum(self.dataset_sizes)*4)
