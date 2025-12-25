# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from torchvision import transforms
import matplotlib.pyplot as plt

__all__ = ["PoetDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens

class PoetDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        # self.tfm_gens = build_transform_gen(cfg, is_train)
        # logging.getLogger(__name__).info(
        #     "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        # )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # @DETR dataset_dict contains 'file_name', 'height', 'width', 'image_id', and 'annotations'
        # it ends with 'file_name', 'height', 'width', 'image_id', 'image', 'instances'
        # dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # file_name = dataset_dict["file_name"]
        # if (file_name == '/home/bmv/poet-detectron/datasets/crab/Datasets/NumPy/image_6.tiff'):
        #     file_name = '/home/bmv/poet-detectron/datasets/crab/Datasets/NumPy/image_5.tiff'
        # npy_file_name = file_name.replace('.tiff', '.npy').replace('_', '')
        # numpy_array = np.load(npy_file_name).astype(np.uint8,copy=True)
        # numpy_array = np.transpose(numpy_array, (1,2,0) ) # (H,W,C)
        # # image = (image*255).astype(np.uint8,copy=True)
        # image= numpy_array[:, : ,0:3]

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        file_path = dataset_dict['file_name']
        dir_path, file_name = file_path.rsplit('/', 1)
        file_name = file_name[::-1].replace('_', '', 1)[::-1]
        image_path = f"{dir_path}/{file_name[:-4]}.npy"
        image = np.load(image_path).astype(np.float64,copy=True) #C H W
        image = image[0:3,:,:] 
        image = image.transpose((1, 2, 0)) # MAA: H W C
        image = (image*255).astype(np.uint8,copy=True) # go to [0, 255]    


        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # utils.check_image_size(dataset_dict, image)

        # if self.crop_gen is None:
        #     image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # else:
        #     if np.random.rand() > 0.5:
        #         image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        #     else:
        #         image, transforms = T.apply_transform_gens(
        #             self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
        #         )

        # image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            from datasets.coco import ConvertCocoPolysToMask, make_coco_transforms
            from PIL import Image
            
            image = Image.fromarray(np.uint8(image)).convert('RGB')

            # add fake area to annotations
            for obj in dataset_dict["annotations"]:
                obj['area'] = -1

            if self.is_train:
                apply_transforms = make_coco_transforms('train', downsample=False) # make transforms
            else:
                apply_transforms = make_coco_transforms('val', downsample=False)

            # @ben center of mass should be decided by kpts_center in config
            image, target = ConvertCocoPolysToMask('center_of_mass')(image, dataset_dict) # prepare image #MAA: returns image size (1086,1030) 
            image, target = apply_transforms(image, target)


            target['height'], target['width'] = target['size']
            target["image"] = image

        return target

