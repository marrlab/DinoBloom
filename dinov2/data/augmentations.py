# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Callable

from torchvision import transforms
from skimage.util import dtype, dtype_limits
from PIL import Image
from skimage.exposure import rescale_intensity
import numpy as np

from .transforms import GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov2")

class HEDJitter(Callable):
    # HED color augmentations 
    # adapted from  https://github.com/DIAGNijmegen/pathology-he-auto-augment/blob/main/he-randaugment/custom_hed_transform.py
    def __init__(self, factor=0.07):
        self.factor = factor
    
        self.rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                            [0.07, 0.99, 0.11],
                            [0.27, 0.57, 0.78]]).astype('float32')

        self.hed_from_rgb = np.linalg.inv(self.rgb_from_hed).astype('float32')

    def rgb2hed(self, rgb):
        rgb = dtype.img_as_float(rgb, force_copy=True).astype('float32')
        rgb += 2
        stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), self.hed_from_rgb)
        return np.reshape(stains, rgb.shape)

    def hed2rgb(self, hed):
        stains = dtype.img_as_float(hed.astype('float64')).astype('float32')  # stains are out of range [-1, 1] so dtype.img_as_float complains if not float64
        logrgb2 = np.dot(-np.reshape(stains, (-1, 3)), self.rgb_from_hed)
        rgb2 = np.exp(logrgb2)
        return rescale_intensity(np.reshape(rgb2 - 2, stains.shape),
                                in_range=(-1, 1))

    def __call__(self, patch):
        __cutoff_range = (0.15, 0.85)
        __biases = [np.random.uniform(-self.factor, self.factor) for _ in range(3)]
        __sigmas = [np.random.uniform(-self.factor, self.factor) for _ in range(3)]
        
        patch_hed = self.rgb2hed(np.array(patch))

        patch_hed *= (1.0 + np.array(__sigmas))
        patch_hed += np.array(__biases)
        
        patch_rgb = self.hed2rgb(hed=patch_hed)
        patch_rgb = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)
        patch_rgb *= 255.0
        patch_rgb = patch_rgb.astype(dtype=np.uint8)
        
        return Image.fromarray(patch_rgb)
            

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                # transforms.CenterCrop(global_crops_size),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # additional histopathology-specific augmentations
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),

            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                # transforms.CenterCrop(global_crops_size),
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                # additional histopathology-specific augmentations
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
        #        transforms.RandomApply(
        #            [HEDJitter(factor=0.07)],
        #            p=0.5,
        #        ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,  # original p=0.8
                ),
                # additional histopathology-specific augmentations (don't use grayscale)
                # transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=0.1)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                # transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
