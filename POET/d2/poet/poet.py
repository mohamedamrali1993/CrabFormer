# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks, Keypoints
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss
from POET.models.backbone import Joiner
from POET.models.matcher import HungarianMatcher
from POET.models.position_encoding import PositionEmbeddingSine
from POET.models.transformer import Transformer
from POET.models.poet import POET, SetCriterion, PostProcess
from POET.util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from POET.util.misc import NestedTensor
from POET.datasets.coco import convert_coco_poly_to_mask

__all__ = ["Poet"]


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class Poet(nn.Module):
    """
    Implement Poet
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.POET.NUM_CLASSES
        hidden_dim = cfg.MODEL.POET.HIDDEN_DIM
        num_queries = cfg.MODEL.POET.NUM_OBJECT_QUERIES
        # Transformer parameters:
        nheads = cfg.MODEL.POET.NHEADS
        dropout = cfg.MODEL.POET.DROPOUT
        dim_feedforward = cfg.MODEL.POET.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.POET.ENC_LAYERS
        dec_layers = cfg.MODEL.POET.DEC_LAYERS
        pre_norm = cfg.MODEL.POET.PRE_NORM

        # Loss parameters:
        # giou_weight = cfg.MODEL.POET.GIOU_WEIGHT
        # l1_weight = cfg.MODEL.POET.L1_WEIGHT
        deep_supervision = cfg.MODEL.POET.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.POET.NO_OBJECT_WEIGHT

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels


        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        # @ben this derives from models/poet.py > build method
        self.poet = POET(
            backbone, transformer, num_classes=self.num_classes, num_queries=num_queries, aux_loss=deep_supervision
        )

        self.poet.to(self.device)


        # building criterion
        matcher = HungarianMatcher(
            cost_class = cfg.MODEL.POET.CLASS, 
            cost_kpts = cfg.MODEL.POET.ABSOLUT_KEYPOINTS,
            cost_ctrs = cfg.MODEL.POET.CTRS,
            cost_deltas = cfg.MODEL.POET.DELTAS,
            cost_kpts_class = cfg.MODEL.POET.KPTS_CLASS,
        )

        self.post_processor= PostProcess() #converts the model's output into the format expected by the coco api
        weight_dict = {
            'loss_ce': cfg.MODEL.POET.CLASS,
            'loss_kpts': cfg.MODEL.POET.ABSOLUT_KEYPOINTS, 
            'loss_ctrs': cfg.MODEL.POET.CTRS, 
            'loss_deltas': cfg.MODEL.POET.DELTAS, 
            'loss_kpts_class': cfg.MODEL.POET.KPTS_CLASS,
        }
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        # losses = ['labels', 'keypoints', 'cardinality'] 
        losses = ['labels', 'keypoints'] #MAA: removed cardinality becuase it does not use gradient and doesn't affect the training
        self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(len(cfg.MODEL.PIXEL_MEAN), 1, 1) #MAA
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(len(cfg.MODEL.PIXEL_STD), 1, 1) #MAA
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        output = self.poet(images)

        if self.training:
            targets = self.prepare_targets(batched_inputs)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            # keys: pred_logits, pred_kpts, aux_outputs

            size= torch.tensor(images.image_sizes, device='cuda:0')
            processed_outputs= self.post_processor(output,  size)
            kpts_cls= [entry['labels'] for entry in processed_outputs]
            kpts_pred= [entry['keypoints'] for entry in processed_outputs]
            kpts_scores= [entry['scores'] for entry in processed_outputs]
            
            results = self.inference(kpts_cls, kpts_pred, kpts_scores,images.image_sizes)
            processed_results = []
            #FS: Compare here to DETR
            for results_per_image in results:
                processed_results.append({"instances": results_per_image}) #directly append the results to the processed_results list without any further processing
            
            return processed_results

    def prepare_targets(self, targets): #FS:takes the input targets (which include labels and keypoints) and organizes them into a new list of dictionaries.
        new_targets = []
        for target_dict in targets:
            new_kpt = target_dict['keypoints'].to(self.device)
            new_target_dict = {
                "labels": target_dict["labels"].to(self.device),
                "keypoints": new_kpt,
            }
            new_targets.append(new_target_dict)

        return new_targets

    def inference(self, kpts_cls, kpts_pred, kpts_scores, image_sizes): #ensures that the raw model outputs are transformed into a meaningful format that can be used for downstream
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(kpts_cls) == len(image_sizes)
        results = []
        
        for i, (scores_per_image, labels_per_image, kpts_pred_per_image, image_size) in enumerate(zip(
            kpts_scores, kpts_cls, kpts_pred, image_sizes
        )):
            result = Instances(image_size)
            # result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            # result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.pred_keypoints = kpts_pred_per_image.reshape(-1, 3, 3) #reshape keypoint predictions to 3x3
            result.pred_boxes= Boxes(tensor=torch.zeros(kpts_pred_per_image.shape[0],4)) #FS: dummy boxes
            result.scores = scores_per_image #FS: scores are the confidence of the keypoints
            result.pred_classes = labels_per_image #FS: class labels
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs): #responsible for normalizing, padding, and batching the input images
        """
        Normalize, pad and batch the input images.
        """
        # images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images) #create ImageList object (ensures padding and batching)
        return images
