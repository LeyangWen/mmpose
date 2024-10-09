# Copyright (c) OpenMMLab. All rights reserved.
from .coco_wholebody_dataset import CocoWholeBodyDataset
from .halpe_dataset import HalpeDataset
from .ubody2d_dataset import UBody2dDataset
from .VEHS7M37kpts_dataset import VEHS7M37kptsDataset

__all__ = ['CocoWholeBodyDataset', 'HalpeDataset', 'UBody2dDataset', 'VEHS7M37kptsDataset']
