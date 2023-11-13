# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
_Target = int


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"  # NOTE: torchvision does not support the test split

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 1_281_167,
            _Split.VAL: 50_000,
            _Split.TEST: 100_000,
        }
        return split_lengths[self]

    def get_dirname(self, class_id: Optional[str] = None) -> str:
        return self.value if class_id is None else os.path.join(self.value, class_id)

    def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
        dirname = self.get_dirname(class_id)
        if self == _Split.TRAIN:
            basename = f"{class_id}_{actual_index}"
        else:  # self in (_Split.VAL, _Split.TEST):
            basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
        return os.path.join(dirname, basename + ".JPEG")

    def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
        assert self != _Split.TEST
        dirname, filename = os.path.split(image_relpath)
        class_id = os.path.split(dirname)[-1]
        basename, _ = os.path.splitext(filename)
        actual_index = int(basename.split("_")[-1])
        return class_id, actual_index


class WSIDataset(VisionDataset):
    Target = Union[_Target]
    Split = Union[_Split]

    def __init__(
        self,
        *,
        root: "/lustre/groups/shared/tcga/CRC/slides",
        load: bool = True,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        if load:
            self.slides = []
            for slide in Path(root).glob("*.svs"):
                slide = openslide.open_slide(str(slide))
                self.slides.append(slide)
        else:
            self.slides = list(Path(root).glob("*.svs"))
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        slide = self.slides[index]

        dim_x, dim_y = slide.dimensions

    def __len__(self) -> int:
        entries = self._get_entries()
        assert len(entries) == self.split.length
        return len(entries)