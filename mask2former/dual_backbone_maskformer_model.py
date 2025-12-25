# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

#MAA Keypoint post process
from POET.models.poet import PostProcess
#MAA Visualization
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TF

@META_ARCH_REGISTRY.register() #register class in the meta architecture registry of Detectron2
class DualBackboneMaskFormer(nn.Module): # MAA: Late Fusion of RGB and Depth features
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        backbone_depth: Backbone, #MAA
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        pixel_mean_depth: Tuple[float], #MAA
        pixel_std_depth: Tuple[float], #MAA
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone #sets the backbone
        self.detph_backbone = backbone_depth #MAA
        self.sem_seg_head = sem_seg_head #sets the semantic segmentation head module
        self.criterion = criterion #set the criterion (loss function) module
        self.num_queries = num_queries #set number of queries
        self.overlap_threshold = overlap_threshold #sets the overlap threshold used in inference
        self.object_mask_threshold = object_mask_threshold #sets the treshold for filtering queries based on classification score
        self.metadata = metadata #sets metadata


        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility #Ensures input height and width are divisible by a specific integer, set based on the backbone if not provided
        self.size_divisibility = size_divisibility 
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference #Determines if resizing predictions before inference is needed.
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False) #register pixel mean buffer for normalization
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False) #register pixel std buffer for normalization
        self.register_buffer("pixel_mean_depth", torch.Tensor(pixel_mean_depth).view(-1, 1, 1), False) #MAA
        self.register_buffer("pixel_std_depth", torch.Tensor(pixel_std_depth).view(-1, 1, 1), False) #MAA

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.post_processor= PostProcess() #converts the model's output into the format expected by the coco api

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        #MAA Backbones
        backbone = build_backbone(cfg) #Constructs the backbone module using the configuration.
        depth_cfg = cfg.clone()
        depth_cfg.MODEL.PIXEL_MEAN = [12.62195593] #MAA: Depth channel mean
        depth_cfg.MODEL.PIXEL_STD = [15.38459126] # MAA: Depth channel std
        backbone_depth = build_backbone(depth_cfg)

        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape()) #Constructs the semantic segmentation head module using the configuration and the output shape of the backbone.

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION   #Determines if deep supervision is used
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT  #Sets the weight for no object

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher( #Creates a HungarianMatcher object to compute the matching cost between predictions and targets
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points= cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS, 
            cost_kpts = cfg.MODEL.POET.ABSOLUT_KEYPOINTS,
            cost_ctrs = cfg.MODEL.POET.CTRS,
            cost_deltas = cfg.MODEL.POET.DELTAS,
            cost_kpts_class = cfg.MODEL.POET.KPTS_CLASS,
        )

        weight_dict = {"loss_ce": class_weight, 
                       "loss_mask": mask_weight, 
                       "loss_dice": dice_weight,
                       "loss_ce": cfg.MODEL.POET.CLASS,
                        "loss_kpts": cfg.MODEL.POET.ABSOLUT_KEYPOINTS, 
                        "loss_ctrs": cfg.MODEL.POET.CTRS, 
                        "loss_deltas": cfg.MODEL.POET.DELTAS, 
                        "loss_kpts_class": cfg.MODEL.POET.KPTS_CLASS,} #Dictionary containing weights for each loss type

        if deep_supervision: #If enabled, creates auxiliary weights for intermediate decoder layers
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "keypoints"] #loss types

        criterion = SetCriterion( #Creates a SetCriterion object to compute the loss
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return { #returns the configuration dictionary
            "backbone": backbone,
            "backbone_depth": backbone_depth, #MAA
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "pixel_mean_depth":depth_cfg.MODEL.PIXEL_MEAN,
            "pixel_std_depth":depth_cfg.MODEL.PIXEL_MEAN,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        stragegy = "dual_average" #MAA


        if stragegy == "dual_addition": #MAA
            orig_images = [x["image"].to(self.device) for x in batched_inputs] 
            
            images = [(x[:3, :, :] - self.pixel_mean) / self.pixel_std for x in orig_images] #normalize images
            images = ImageList.from_tensors(images, self.size_divisibility) #Packs images into an ImageList object, ensuring they meet the size divisibility requirement.
            features = self.backbone(images.tensor) #extract features from the backbone

            depth_maps = [(x[3:, :, :] - self.pixel_mean_depth) / self.pixel_std_depth for x in orig_images] #normalize images
            depth_maps = ImageList.from_tensors(depth_maps, self.size_divisibility) #Packs images into an ImageList object, ensuring they meet the size divisibility requirement.
            features_depth = self.detph_backbone(depth_maps.tensor) #extract features from the backbone

            # self.main_visualization(orig_images, features, features_depth)

            final_features = {key: features[key] + features_depth[key] for key in features.keys()}

            outputs = self.sem_seg_head(final_features) #get semantic segmentation outputs
        elif stragegy == "dual_average": #MAA
            orig_images = [x["image"].to(self.device) for x in batched_inputs]  
            images = [(x[:3, :, :] - self.pixel_mean) / self.pixel_std for x in orig_images] #normalize images
            images = ImageList.from_tensors(images, self.size_divisibility) #Packs images into an ImageList object, ensuring they meet the size divisibility requirement.

            features = self.backbone(images.tensor) #extract features from the backbone

            depth_maps = [(x[3:, :, :] - self.pixel_mean_depth) / self.pixel_std_depth for x in orig_images] #normalize images
            depth_maps = ImageList.from_tensors(depth_maps, self.size_divisibility) #Packs images into an ImageList object, ensuring they meet the size divisibility requirement.
            
            features_depth = self.detph_backbone(depth_maps.tensor) #extract features from the backbone

            # self.main_visualization(orig_images, features, features_depth)

            final_features = {key: (features[key] + features_depth[key])/2 for key in features.keys()}

            outputs = self.sem_seg_head(final_features) #get semantic segmentation outputs
        elif stragegy == "dual_concatenation": #MAA: Does not work
            class FeatureEmbed(nn.Module): #added by FS to embbed the features
                    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
                        super().__init__()
                        self.in_chans = in_chans
                        self.embed_dim = embed_dim

                        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1)

                        # Explicitly set weights and bias to float32
                        self.proj.weight = nn.Parameter(self.proj.weight.float())
                        if self.proj.bias is not None:
                            self.proj.bias = nn.Parameter(self.proj.bias.float())

                        if norm_layer is not None:
                            self.norm = norm_layer(embed_dim)
                        else:
                            self.norm = None

                        self.float()

                    def forward(self, x):
                        """Forward function.""" 
                        # Process feature map with convolution
                        x = x.float()
                        x = self.proj(x) 
                        return x

            orig_images = [x["image"].to(self.device) for x in batched_inputs]  
            images = [(x[:3, :, :] - self.pixel_mean) / self.pixel_std for x in orig_images] #normalize images
            images = ImageList.from_tensors(images, self.size_divisibility) #Packs images into an ImageList object, ensuring they meet the size divisibility requirement.

            features = self.backbone(images.tensor) #extract features from the backbone

            depth_maps = [(x[3:, :, :] - self.pixel_mean_depth) / self.pixel_std_depth for x in orig_images] #normalize images
            depth_maps = ImageList.from_tensors(depth_maps, self.size_divisibility) #Packs images into an ImageList object, ensuring they meet the size divisibility requirement.
            
            features_depth = self.detph_backbone(depth_maps.tensor) #extract features from the backbone
            # Instantiate embedding layers for both RGB and depth
            rgb_embed = FeatureEmbed(in_chans=features[list(features.keys())[0]].shape[1], embed_dim=96, norm_layer=None)
            depth_embed = FeatureEmbed(in_chans=features_depth[list(features_depth.keys())[0]].shape[1], embed_dim=96, norm_layer=None)

                # Apply the embedding to each feature map in both dictionaries
            features = {key: rgb_embed(value) for key, value in features.items()}
            features_depth = {key: depth_embed(value) for key, value in features_depth.items()}

            reduce_channels = {}
            final_features = {key: torch.cat((features[key], features_depth[key]), dim=1) for key in features.keys()}
            for key, value in final_features.items():
                reduce_channels[key] = nn.Sequential(
                    nn.Conv2d(value.shape[1], value.shape[1] // 2, kernel_size=1, stride = 1, padding = 0),
                    nn.GroupNorm(32, value.shape[1] // 2)
                ).to(self.device)
            final_features = {key: reduce_channels[key](value.to(self.device)) for key, value in final_features.items()}
            outputs = self.sem_seg_head(final_features) #get semantic segmentation outputs
        else:
            raise ValueError("Invalid strategy")

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_keypoints = [x["kpts"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images, gt_keypoints) #Converts ground truth instances to target format for loss calculation
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets) #compute loss with criterion

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k] #apply weights to losses
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses #returns the loses for backpropagation 
        else: #inference mode

            # kpts_pred_results = outputs["pred_kpts"] #get keypoint prediction results
            size= torch.tensor(images.image_sizes, device='cuda:0')
            processed_outputs= self.post_processor(outputs, size)

            kpts_pred_results= [instance['keypoints'] for instance in processed_outputs]
            mask_cls_results = outputs["pred_logits"] #get mask classification results (logits)
            mask_pred_results = outputs["pred_masks"] #get mask prediction results

            # upsample masks
            mask_pred_results = F.interpolate( #Resizes predicted masks to match the input image size using bilinear interpolation.
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = [] #initialize list to store processed results
            for mask_cls_result, mask_pred_result, kpts_pred_result, input_per_image, image_size in zip( #Iterates over the batch to process predictions for each image.
                mask_cls_results, mask_pred_results, kpts_pred_results, batched_inputs, images.image_sizes
            ):
                #get the height and width for output resizing
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference: #Optionally resizes mask predictions before inference.
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, kpts_pred_result)
                    processed_results[-1]["instances"] = instance_r


            return processed_results

    #MAA visualization of feature maps
    def visualize_features(self, features, title):
        """
        Visualize the feature maps.
        """
        fig, axs = plt.subplots(1, len(features), figsize=(20, 5))
        fig.suptitle(title, fontsize=16)
        for i, (name, feature) in enumerate(features.items()):
            # Normalize each feature map to [0, 1] for visualization
            feature = feature[0]
            feature = (feature - feature.min()) / (feature.max() - feature.min())
            
            feature = torch.mean(feature, dim=0, keepdim=True)  # Take the mean across channels
            feature = feature.cpu().detach().numpy().transpose(1, 2, 0)
            axs[i].imshow(feature, cmap='jet')
            axs[i].set_title(name)
            axs[i].axis('off')

    def main_visualization(self, orig_images, features, features_depth):
        # Visualize original images
        for i, orig_image in enumerate(orig_images):
            plt.figure(figsize=(12, 8))
            
            # Original BGR image
            plt.subplot(1, 2, 1)
            bgr_image = orig_image[:3, :, :].cpu().numpy().transpose(1, 2, 0)
            rgb_image = bgr_image[:,:,::-1]
            rgb_image = (rgb_image).astype(np.float32) /255
            plt.imshow(rgb_image)
            plt.title('Original RGB Image')
            plt.axis('off')

            # Depth map
            plt.subplot(1, 2, 2)
            depth_map = orig_image[3:, :, :].cpu().numpy().squeeze()
            plt.imshow(depth_map, cmap='jet')
            plt.title('Depth Map')
            plt.axis('off')

            # Visualize features
            self.visualize_features(features, "RGB Backbone Features")
            self.visualize_features(features_depth, "Depth Backbone Features")
            plt.show()
            exit()



    def prepare_targets(self, targets, images, keypoints):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image, keypoints_targets in zip(targets, keypoints):
            # pad gt
            gt_masks = targets_per_image.gt_masks.tensor
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            # new_kpt = targets_per_image['keypoints'].to(self.device) #original keypoints
            new_kpt = keypoints_targets.to(self.device) #original keypoints
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "keypoints": new_kpt
                }
            )
        # import pdb; pdb.set_trace()
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, kpts_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        result.pred_keypoints = kpts_pred.reshape(-1, 3, 3)[topk_indices]

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        return result
