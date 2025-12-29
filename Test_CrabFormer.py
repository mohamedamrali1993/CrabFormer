#!/usr/bin/env python3
"""
CrabFormer Inference Script.

"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass


import os
import time
import cv2
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
    DatasetCatalog,
)
from detectron2.engine import (
    DefaultPredictor,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.engine.hooks import HookBase
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.config import  get_cfg

# Mask2Former imports
from mask2former import (
    add_maskformer2_config,
    add_dual_swin_maskformer2_config
)
# POET imports
from POET.d2.poet import add_poet_config

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

class CrabLoadingPredictor(DefaultPredictor):
    
    def __init__(self, *args, **kwargs):
        super(CrabLoadingPredictor, self).__init__(*args, **kwargs)

    def __call__(self, image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.

            height, width = image.shape[:2]
            # image = self.aug.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)) #C W H

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

def prioritize(outputs, rgb, depth):
    # priority based on depth map top instance

    instances= outputs
    instances_num= len(instances)
    # boxes= instances.pred_boxes.tensor
    scores= instances.scores
    classes= instances.pred_classes
    keypoints= instances.pred_keypoints
    masks= instances.pred_masks

    
    average_list=[]
    for i in range(masks.shape[0]):
        mask_instance= masks[i].numpy()
        # instance_depth= np.multiply(depth,mask_instance)
        mask_instance_exp= np.expand_dims(mask_instance,axis=0) 
        height= np.multiply(mask_instance_exp, depth)
        # mask_average= np.mean(depth[mask_instance_exp])
        mask_average= np.mean(height)
        average_list.append(mask_average)
        
    average_list_np= np.array(average_list)
    priority_indicies= np.argsort(average_list_np)[::-1]
    print(priority_indicies)
    


    target= priority_indicies[0]
    chosen_mask= masks[target].numpy()
    chosen_keypoints = keypoints[target].numpy()
    plt.figure(1)
    plt.imshow(chosen_mask)
    for kp in chosen_keypoints:
        x, y, confidence = kp  # (x, y, confidence)
        if confidence > 0:  # Only show keypoints with a valid confidence score
            plt.scatter(x, y, s=50, c='red', marker='o', edgecolors='black')  # Circle at (x, y)
    
    keypoint_1 = chosen_keypoints[0][:2]  # Coordinates of keypoint 1
    keypoint_2 = chosen_keypoints[1][:2]  # Coordinates of keypoint 2
    keypoint_3 = chosen_keypoints[2][:2]  # Coordinates of keypoint 3
    midpoint_13 = (keypoint_1 + keypoint_3) / 2  # Midpoint of keypoint 1 and keypoint 3
    vector_mid_to_2 = keypoint_2 - midpoint_13

    plt.scatter(midpoint_13[0], midpoint_13[1], s=100, c='blue', marker='x', edgecolors='black', label="Midpoint (1,3)", linewidths=2)
    plt.arrow(midpoint_13[0], midpoint_13[1], vector_mid_to_2[0], vector_mid_to_2[1],
              head_width=50, head_length=30, fc='green', ec='green', label="Vector Midpoint to 2")

    # from matplotlib.patches import FancyArrowPatch
    # arrow = FancyArrowPatch(posA=midpoint_13, posB=keypoint_2, arrowstyle='->,head_width=7.0,head_length=7.0',
    #                         color='green', lw=2)
    # plt.gca().add_patch(arrow)  # Add the custom arrow to the plot


    plt.xticks([])  # Remove x ticks
    plt.yticks([])  # Remove y ticks
    plt.show(block= False)

    # for i in enumerate(instances.pred_masks):
    #     plt.figure(i)
    #     plt.imshow(instances.pred_masks[i])
    # plt.show(block = True)

    return chosen_mask

def test(cfg, dataset , mode=False, infrence_num=  1):
    if mode == True:        
        predictor = CrabLoadingPredictor(cfg)

        # for d in random.sample(dataset, infrence_num): #Numbering: [1,10-19,2,20-25,3-9]   labeled indicies:0-10, unlabelled indicies:11-25
        for d in dataset[20:25]: #Numbering: [1,10-19,2,20-25,3-9]   labeled indicies:0-10, unlabelled indicies:11-25
            print("{}".format(d["file_name"]))
            directory, filename = os.path.split(d["file_name"])
            modified_filename = filename.replace('_', '')[:-5] + '.npy'
            modified_path = os.path.join(directory, modified_filename)
            image = np.load(modified_path)
            image = image[:,:,:] #C H W
            depth= image[3:,:,:]
            image = image.transpose((1, 2, 0)) # go to W H C
            image = (image).astype(np.float32) # go to [0, 255]
            

            t1= time.perf_counter()
            outputs = predictor(image)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            print('model ran in: {}'.format(time.perf_counter()-t1))

            instances = outputs["instances"].to("cpu")
            instances.remove("pred_boxes") 
            # import pdb; pdb.set_trace()
            high_score_mask = instances.scores >= 0.10
            filtered_instances = instances[high_score_mask]
            size_threshold = 500000
            small_mask_indices = filtered_instances.pred_masks.sum(dim=(2,1)) < size_threshold
            filtered_instances = filtered_instances[small_mask_indices]

            # u, v, top_mask= prioritize(outputs,image)
            # import pdb; pdb.set_trace()
            image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB )#H W C
            brightness_factor =1.5
            image_bright = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
            v = Visualizer(image_bright[:,:,::-1], metadata= MetadataCatalog.get(cfg.DATASETS.TEST[1]), scale=1)
            out = v.draw_instance_predictions(filtered_instances)

            prioritize(filtered_instances, image_bright, depth)

            rgb_depth_result_imshow(out.get_image(), image_bright[:, :,::-1], depth_image= depth)

def main(args):
    cfg = setup(args)
    # MAA: Testing the model and previewing results for visualization
    test(cfg=cfg, dataset= crab_pile_val_dataset_dicts, mode= args.eval_only, infrence_num=1)

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


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    ##############################################################################################################################################
    # MAA: Regular Configurations
    # args.config_file = "Mask2Former/configs/BMV_coco/maskformer2_R50_bs16_50ep.yaml" #trains with ResNet50 RGBD input
    args.config_file = "Mask2Former/configs/BMV_coco/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml" #trains with SWIN-Large RGBD input
    # args.config_file = "CrabFormer/configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml" #SWIN-Tiny RGBD input

    # MAA: Dual Backbone Configurations
    # args.config_file = "CrabFormer/configs/coco/instance-segmentation/swin/MAA/maskformer2_dual_swin_tiny.yaml" #SWIN-Tiny Dual Backbone
    
    #MAA: Dual Patch Configurations
    # args.config_file = "CrabFormer/configs/coco/instance-segmentation/swin/MAA/maskformer2_dual_patch_swin_tiny.yaml" #SWIN-Tiny Dual Patch
    ###########################################################################################################################################
    
    #MAA: Test time model weights (must have the same configuration as the training model)
    args.opts = ["MODEL.WEIGHTS", "Path/to/your/model_checkpoint.pth"]  # path to the model we just trained
    args.eval_only = True
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
