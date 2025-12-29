#!/usr/bin/env python3
"""
CrabFormer Training & Eval Script.

"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import time
import datetime
import cv2
from collections import OrderedDict
from typing import Any, Dict, List, Set
from PIL import Image
import numpy as np
from colorama import Fore, Style, init
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch, torchvision

# detectron2 imports
from detectron2.utils.visualizer import Visualizer, GenericMask
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
    DatasetCatalog,
)
from detectron2.engine import (
    DefaultTrainer,
    DefaultPredictor,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger, log_every_n_seconds
from detectron2.data.datasets import register_coco_instances
from detectron2.config import  get_cfg

# Mask2Former imports
from mask2former import (
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    add_dual_swin_maskformer2_config
)
# POET imports
from POET.d2.poet import add_poet_config

from depth_augmentations import add_gaussian_noise, depth_shift, elastic_distortion, apply_gaussian_blur, cutout, dropout, jitter

# Register Crab Loading Datasets
dataset_names= [
        "crab_easy_train",
        "crab_easy_train_2",
        "crab_easy_val",
        "crab_medium_train",
        "crab_medium_train_2",
        "crab_medium_val",
        "crab_pile_train",
        "crab_pile_train_2",
        "crab_pile_val",
        "crab_pile_val_top_instance", # MAA: Validation Set for Top Instance Only
        "crab_pile_val_core", # MAA: Validation Set for Core of all instances
        "crab_pile_val_core_top_instance" # MAA: Validation Set for Core of Top Instance Only
    ]
json_file_paths = [
        "CrabFormer/labels/Easy_case_train_intact.json",
        "CrabFormer/labels/Easy_case_train_intact_2.json",
        "CrabFormer/labels/Easy_case_val_intact.json",
        "CrabFormer/labels/Medium_case_train_intact.json",
        "CrabFormer/labels/Medium_case_train_intact_2.json",
        "CrabFormer/labels/Medium_case_val_intact.json",
        "CrabFormer/labels/Piled_case_train_intact.json",
        "CrabFormer/labels/Piled_case_train_intact_2.json",
        "CrabFormer/labels/Piled_case_val_intact.json",
        "CrabFormer/labels/Piled_case_val_intact_top_instance.json",
        "CrabFormer/labels/Piled_case_val_intact_core.json",
        "CrabFormer/labels/Piled_case_val_intact_core_top_instance.json",
    ]
image_roots = [
        "CrabFormer/Datasets/",
    ]

# register with detectron2 => get json data (calls empty annotations)
for i in range(len(dataset_names)):
    register_coco_instances(dataset_names[i], {}, json_file_paths[i], image_roots[0])
    MetadataCatalog.get(dataset_names[i]).thing_classes= ['Intact-belly_down','Intact-belly_up']
    MetadataCatalog.get(dataset_names[i]).keypoint_names = ["rear_right", "front_center","rear_left"]
    MetadataCatalog.get(dataset_names[i]).keypoint_connection_rules = [("rear_right","front_center",(255,0,0)), ("rear_left","front_center",(0,255,0))]
    MetadataCatalog.get(dataset_names[i]).keypoint_flip_map= [("rear_right","rear_left")]
    print("Registered Dataset: ", dataset_names[i])


crab_easy_train_metadata = MetadataCatalog.get(dataset_names[0])
crab_easy_train_dataset_dicts = DatasetCatalog.get(dataset_names[0])
crab_easy_train_2_metadata = MetadataCatalog.get(dataset_names[1])
crab_easy_train_2_dataset_dicts = DatasetCatalog.get(dataset_names[1])
crab_medium_train_metadata = MetadataCatalog.get(dataset_names[3])
crab_medium_train_dataset_dicts = DatasetCatalog.get(dataset_names[3])
crab_medium_train_2_metadata = MetadataCatalog.get(dataset_names[4])
crab_medium_train_2_dataset_dicts = DatasetCatalog.get(dataset_names[4])
crab_pile_train_metadata = MetadataCatalog.get(dataset_names[6])
crab_pile_train_dataset_dicts = DatasetCatalog.get(dataset_names[6])
crab_pile_train_2_metadata = MetadataCatalog.get(dataset_names[7])
crab_pile_train_2_dataset_dicts = DatasetCatalog.get(dataset_names[7])

crab_easy_val_metadata = MetadataCatalog.get(dataset_names[2])
crab_easy_val_dataset_dicts = DatasetCatalog.get(dataset_names[2])
crab_medium_val_metadata = MetadataCatalog.get(dataset_names[5])
crab_medium_val_dataset_dicts = DatasetCatalog.get(dataset_names[5])
crab_pile_val_metadata = MetadataCatalog.get(dataset_names[8])
crab_pile_val_dataset_dicts = DatasetCatalog.get(dataset_names[8])

global augmentation
augmentation = True #MAA: Training Augmentation control!
init()
def print_colored(text, color):
    colors = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'blue': Fore.BLUE,
        'yellow': Fore.YELLOW,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
    }
    reset = Style.RESET_ALL
    print(colors.get(color, Fore.WHITE) + text + reset)

class CrabLoadingInstanceMapper:
    def __init__(
        self,
        tfm_gens=[],
        is_train=True,
    ):
        self.is_train = is_train
        self.tfm_gens = tfm_gens
        # self.img_format = image_format

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {tfm_gens}")

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        directory, filename = os.path.split(dataset_dict["file_name"])
        modified_filename = filename.replace('_', '')[:-5] + '.npy'
        modified_path = os.path.join(directory, modified_filename)
        image = np.load(modified_path)
        rgb = image[0:3,:,:]
        depth= image[3:,:,:] #MAA: Uncomment for depth channel
        image = rgb.transpose((1, 2, 0)) # MAA: H W C
        image = np.nan_to_num(image).astype(np.float32,copy=True)
        depth= depth.transpose((1, 2, 0)) # MAA: H W C MAA: Uncomment for depth channel
        depth = np.nan_to_num(depth).astype(np.float32,copy=True) #MAA: Uncomment for depth channel

        utils.check_image_size(dataset_dict, image)
        
        # Depth Augmentation
        if len(self.tfm_gens) > 2 : #MAA: Uncomment for depth channel
        # if(False): #MAA: comment for depth channel
            original_depth = depth.copy()
            gaussian_noise = add_gaussian_noise(original_depth)
            depth_shifted = depth_shift(gaussian_noise)
            elastic_distorted = elastic_distortion(depth_shifted)
            blurred = apply_gaussian_blur(elastic_distorted)
            cut = cutout(blurred)
            drop = dropout(cut)
            jittered = jitter(drop)
            depth = np.expand_dims(jittered, axis= 2)

        #RGB Augmentation
        augs =T.AugmentationList(self.tfm_gens)
        input = T.AugInput(image)
        transform = augs(input)
        image = input.image
        transforms = input.apply_augmentations(transform) 
        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image

        image= np.concatenate((image, depth), axis=2) #MAA: Uncomment for depth channel

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).to(dtype=torch.float32)
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        # Dataset Dict:  dict_keys(['file_name', 'height', 'width', 'image_id', 'annotations', 'image', 'padding_mask'])
        # Annotations:  (['iscrowd', 'bbox', 'keypoints', 'category_id', 'segmentation', 'bbox_mode'])

        from POET.datasets.coco import ConvertCocoPolysToMask, make_coco_transforms
        image = Image.fromarray(np.uint8(image[:,:,0:3])).convert('RGB')
        for obj in dataset_dict["annotations"]:
            obj['area'] = -1
        
        apply_transforms = make_coco_transforms('train', downsample=False) # make transforms
        poet_image, poet_annos= ConvertCocoPolysToMask('center_of_mass')(image, dataset_dict) # prepare image #MAA: returns image size (1086,1030)
        poet_image, poet_annos = apply_transforms(poet_image, poet_annos)

        dataset_dict['kpts']= poet_annos["keypoints"]
        dataset_dict['labels']= poet_annos["labels"]

        if "annotations" in dataset_dict:
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=utils.create_keypoint_hflip_indices(dataset_names))
                for obj in dataset_dict.pop("annotations")
                if obj["iscrowd"] == 0
            ]

            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)
            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                instances.gt_masks = gt_masks
            dataset_dict["instances"] = instances
            # dataset_dict:   dict_keys(['file_name', 'height', 'width', 'image_id', 'image', 'padding_mask', 'instances'])
        
        

        return dataset_dict
    
class CrabLoadingCOCOEvaluator(COCOEvaluator):
    def __init__(self, *args, **kwargs):
        # super(CrabLoadingCOCOEvaluator, self).__init__(*args, **kwargs)  # ToDo: set tasks?
        super().__init__(*args, **kwargs)  # ToDo: set tasks?

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        import itertools
        from tabulate import tabulate
        from detectron2.utils.logger import create_small_table

        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": [
                "AP",
                "AP50",
                "AP75",
                "APs",
                "APm",
                "APl",
                "AR1",
                "AR10",
                "AR100",
                "ARs",
                "ARm",
                "ARl",
            ],
            "segm": [
                "AP",
                "AP50",
                "AP75",
                "APs",
                "APm",
                "APl",
                "AR1",
                "AR10",
                "AR100",
                "ARs",
                "ARm",
                "ARl",
            ],
            "keypoints": [
                "AP",
                "AP50",
                "AP75",
                "APm",
                "APl",
                "AR",
                "AR50",
                "AR75",
                "ARm",
                "ARl",
            ],  # AP for OKS
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan"
            )
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type)
            + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)]
        )
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP -" + name: ap for name, ap in results_per_category})

        return results

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

class CrabLoadingTrainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        # evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        evaluator_type = "coco"
        if evaluator_type == "coco":
            evaluator_list.append(
                CrabLoadingCOCOEvaluator(dataset_name, output_dir=output_folder, kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS)
            )
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "CocoInstanceMapper":
            global augmentation
            if augmentation == True:
                print_colored(f'Augmentation is True', 'red')
                tfm_gens = [
                        T.RandomBrightness(0.9, 1.2),
                        T.RandomContrast(0.9, 1.2), 
                        T.RandomSaturation(0.9, 1.2),
                        T.RandomLighting(255)
                ]
            else:
                print_colored(f'Augmentation is False', 'red')
                tfm_gens = []
            mapper = CrabLoadingInstanceMapper(is_train= True, tfm_gens= tfm_gens)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):

        return build_detection_test_loader(
            cfg, dataset_name, mapper=CrabLoadingInstanceMapper(is_train= True, tfm_gens= [])
        )

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = (
                        hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                    )
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    # print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model, self.build_test_loader(self.cfg, self.cfg.DATASETS.TEST))) #build_test_loader(cfg, dataset_name)
        return hooks

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.OUTPUT_DIR= "./Path/to/checkpoint/directory" #MAA: RGBD_Q##_  ResNet50, SwinT, SwingL, DualPactchSwinT,  DualSwinT_concat, DualSwinT_add, DualSwinT_mean
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.1,0.1,0.1]
    
    # for poly lr schedule
    add_deeplab_config(cfg)

    ## Adding configurations from config functions
    # If we are doing dual patch SWIN
    if args.config_file == "CrabFormer/configs/coco/instance-segmentation/swin/MAA/maskformer2_dual_patch_swin_tiny.yaml":
        print_colored("Add Dual Swin Configuration for Dual Patches",'red')
        add_dual_swin_maskformer2_config(cfg)
    else:
        add_maskformer2_config(cfg)
    
    add_poet_config(cfg) # add POET config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)


    default_setup(cfg, args)
    cfg.DATASETS.TRAIN = ("crab_easy_train", "crab_easy_train_2" , "crab_medium_train", "crab_medium_train_2" ,"crab_pile_train", "crab_pile_train_2")
    cfg.DATASETS.TEST = ("crab_easy_val", "crab_medium_val", "crab_pile_val", "crab_pile_val_top_instance", "crab_pile_val_core", "crab_pile_val_core_top_instance")


    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    cfg.SOLVER.IMS_PER_BATCH= 1
    cfg.SOLVER.MAX_ITER = 100000
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES= 100 #MAA: 20,30,50,100
    

    # RGB-D Stats
    # Regular model (Early Fusion): takes RGB-D, 4 channels
    # Dual patch (Intermediate Fusion): takes RGB-D, 4 channels
    # Dual backbone (Late Fusion): takes RGB only data, Depth is hard coded
    intact_mean= [44.53766757, 73.17263865, 52.85218037, 12.62195593] #BGRD
    intact_std = [25.17313806, 41.2114908,  21.94960777, 15.38459126] #BGRD

    #MAA: Dual Backbone (Late Fusion) takes RGB only
    if args.config_file == "CrabFormer/configs/coco/instance-segmentation/swin/MAA/maskformer2_dual_swin_tiny.yaml":
        cfg.MODEL.PIXEL_MEAN = intact_mean[0:3]
        cfg.MODEL.PIXEL_STD = intact_std[0:3]
    else:
        cfg.MODEL.PIXEL_MEAN = intact_mean
        cfg.MODEL.PIXEL_STD = intact_std

    cfg.TEST.EVAL_PERIOD = 500
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000

    
    cfg.TEST.DETECTIONS_PER_IMAGE = 15
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.DATALOADER.NUM_WORKERS = 2

    # Setup logger for "mask_former" module
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former"
    )

    return cfg

class GenericMaskPatched(GenericMask):
    def __init__(self, *args, **kwargs):
        super(GenericMaskPatched, self).__init__(*args, **kwargs)
    
    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        # res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # hierarchy = res[-1]
        # if hierarchy is None:  # empty mask
        #     return [], False
        # has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        # res = res[-2]
        # # res = [x.flatten() for x in res]

        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res_orig = res[-2]
        res_holes = [res_orig[i].flatten() for i in range(len(res_orig)) if hierarchy[0][i][3] > -1] # this gives only holes
        res_external = [res_orig[i].flatten() for i in range(len(res_orig)) if hierarchy[0][i][3] == -1] # this gives only external (with holes filled in)
        res = [res_orig[i].flatten() for i in range(len(res_orig))] # this gives holes and external (default)
        
        # if has_holes:
        #     print('HOLEY', len(res_external), len(res_holes), len(res))
        # else:
        #     print('NORMAL', len(res_external), len(res))

        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes, res_external, res_holes
    
    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes, self._external_polygons, self._internal_polygons = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def external_polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes, self._external_polygons, self._internal_polygons = self.mask_to_polygons(self._mask)
        return self._external_polygons

    @property
    def internal_polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes, self._external_polygons, self._internal_polygons = self.mask_to_polygons(self._mask)
        return self._internal_polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes, self._external_polygons, self._internal_polygons = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False  # if original format is polygon, does not have holes
        return self._has_holes

class VisualizerPatched(Visualizer):
    def __init__(self, *args, **kwargs):
        super(VisualizerPatched, self).__init__(*args, **kwargs)
    
    def draw_instance_predictions(self, predictions):
        from detectron2.utils.visualizer import ColorMode, _create_text_labels

        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMaskPatched(x, self.output.height, self.output.width) for x in masks] # GenericMaskPatched instead
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                # self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
    
    def draw_mask(self, mask, color, edge_color=None, alpha=0.5):
        # the difference between this and draw polygons is that this implementation uses path,
        # which can draw holes, whereas polygon cannot.
        """
        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted. If not provided, a darker shade
                of the polygon color will be used instead.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        import matplotlib as mpl
        import matplotlib.colors as mplc
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        # segment = mask.external_polygons[1].reshape(-1, 2)
        # polygon = mpl.patches.Polygon(
        #     segment,
        #     fill=True,
        #     facecolor=mplc.to_rgb(color) + (alpha,),
        #     edgecolor=edge_color,
        #     linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        # )
        # self.output.ax.add_patch(polygon)

        total_path = []
        total_actions = []

        # add polygons function
        def add_polygon(polygons):
            for polygon in polygons:
                polygon = np.array(polygon).reshape(-1, 2)

                if len(polygon > 4):
                    total_path.extend(polygon)
                    actions = [Path.MOVETO] + [Path.LINETO]*(len(polygon)-2) + [Path.CLOSEPOLY]
                    total_actions.extend(actions)

        # add all polygons
        add_polygon(mask.external_polygons)
        add_polygon(mask.internal_polygons)

        try:
            path = Path(total_path, total_actions)
            patch = PathPatch(
                path,
                fill=True,
                facecolor=mplc.to_rgb(color) + (alpha,),
                edgecolor=edge_color,
                linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
            )
            self.output.ax.add_patch(patch)
        except:
            print(len(total_path), 'total_path')
            print(len(total_actions), 'total_actions')

        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5,
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.
        Returns:
            output (VisImage): image object with visualizations.
        """
        from detectron2.utils.colormap import random_color
        from detectron2.utils.visualizer import _SMALL_OBJECT_AREA_THRESH

        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                if len(masks[i].internal_polygons) > 0: # there are holes
                    self.draw_mask(masks[i], color, alpha=alpha)
                else:
                    for segment in masks[i].polygons:
                        self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    # skip small mask without polygon
                    if len(masks[i].polygons) == 0:
                        continue

                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output

def main(args):
    cfg = setup(args)
    train_datasets = cfg.DATASETS.TRAIN
    total_image_number = 0
    for dataset_name in train_datasets:
        dataset_dicts = DatasetCatalog.get(dataset_name)
        total_image_number += len(dataset_dicts)
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    iterations= cfg.SOLVER.MAX_ITER
    epochs= (total_image_number/batch_size)*iterations
    print(f"***********  We are training for {epochs} Epochs  ***********")
    trainer = CrabLoadingTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

def rgb_result_imshow(image, original_image):
    # cv2 BGR image => cv2 RGB image => change range to [0, 1] and make it a tensor => PIL Image and show it
    im1= torchvision.transforms.ToPILImage()(torch.Tensor(np.transpose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255, (2, 0, 1))))
    im2= torchvision.transforms.ToPILImage()(torch.Tensor(np.transpose(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)/255, (2, 0, 1))))
    Image.fromarray(np.hstack((np.array(im1),np.array(im2)))).show()

def rgb_depth_result_imshow(image, original_image, depth_image):
    if len(depth_image.shape) == 3:
            depth_image = np.squeeze(depth_image)  # Squeeze any extra dimension if it's 3D

    im1= torchvision.transforms.ToPILImage()(torch.Tensor(np.transpose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255, (2, 0, 1))))
    im2= torchvision.transforms.ToPILImage()(torch.Tensor(np.transpose(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)/255, (2, 0, 1))))


    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_inverted = 255 - depth_image_normalized  # Invert the depth values

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_inverted), cv2.COLORMAP_JET)

    # Set up the plot to display RGB images and depth colormap
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(im1)
    axs[0].set_title("Image with Visualized Data")
    axs[0].axis('off')

    axs[1].imshow(im2)
    axs[1].set_title("Original Image")
    axs[1].axis('off')

    # Show the depth image with jet colormap and color bar
    depth_display = axs[2].imshow(depth_colormap)
    axs[2].set_title("Depth Image")
    axs[2].axis('off')

    norm = mpl.colors.Normalize(vmin=np.min(depth_image), vmax=np.max(depth_image))
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])  # Required for colorbar

    cbar = fig.colorbar(sm, ax=axs[2], fraction=0.046, pad=0.04)
    cbar.set_label('Depth (mm)', rotation=90, labelpad=20)  
    

    # Display the plot
    plt.show()

def preview_dataset():
    num_images = 1
    for d in crab_pile_train_2_dataset_dicts[0:10]:
        print("{}".format(d["file_name"]))
        directory, filename = os.path.split(d["file_name"])
        modified_filename = filename.replace('_', '')[:-5] + '.npy'
        modified_path = os.path.join(directory, modified_filename)
        image = np.load(modified_path)
        rgb = image[0:3,:,:] 
        depth = image[3:,:,:]
        image = rgb.transpose((1, 2, 0)) # go to W H C # MAA: H W C
        image = (image).astype(np.uint8,copy=True) # go to [0, 255]
        # Increase brightness by multiplying with a brightness factor
        brightness_factor = 1.5  # Adjust this factor for desired brightness
        image_bright = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        visualizer = Visualizer(image_bright, metadata= crab_easy_val_metadata, scale=1)
        out = visualizer.draw_dataset_dict(d)
        # rgb_result_imshow(out.get_image(),image_bright)
        rgb_depth_result_imshow(out.get_image(),image_bright, depth)

if __name__ == "__main__":
    # preview_dataset() # Preview dataset images with annotations for debugging.
    args = default_argument_parser().parse_args()

    ##############################################################################################################################################
    # MAA: Regular Configurations
    # args.config_file = "CrabFormer/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml" #trains with ResNet50 RGBD input
    args.config_file = "CrabFormer/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml" #trains with SWIN-Large RGBD input
    # args.config_file = "CrabFormer/configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml" #SWIN-Tiny RGBD input

    # MAA: Dual Backbone Configurations
    # args.config_file = "CrabFormer/configs/coco/instance-segmentation/swin/MAA/maskformer2_dual_swin_tiny.yaml" #SWIN-Tiny Dual Backbone
    
    #MAA: Dual Patch Configurations
    # args.config_file = "CrabFormer/configs/coco/instance-segmentation/swin/MAA/maskformer2_dual_patch_swin_tiny.yaml" #SWIN-Tiny Dual Patch
    ###########################################################################################################################################
    
    # Initial Training Weights
    args.opts = ["MODEL.WEIGHTS", ""] #Training from scratch
    # args.opts = ["MODEL.WEIGHTS", "Mask2Former/mask2former/modeling/backbone/swin_large_patch4_window12_384_22k.pkl"]
    
    
    args.resume = False
    args.num_gpus = 1

    # print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
