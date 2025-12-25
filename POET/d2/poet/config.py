# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_poet_config(cfg):
    """
    Add config for POET.
    """
    cfg.MODEL.POET = CN()
    cfg.MODEL.POET.NUM_CLASSES = 2

    # For Segmentation
    cfg.MODEL.POET.FROZEN_WEIGHTS = ''

    # LOSS
    cfg.MODEL.POET.DEEP_SUPERVISION = True
    cfg.MODEL.POET.NO_OBJECT_WEIGHT = 0.1 #MAA: Potentially change this

    #MAA: Potentially change this
    cfg.MODEL.POET.CLASS = 1 # Object class coefficient
    cfg.MODEL.POET.ABSOLUT_KEYPOINTS = 4 # Object absolute keypoints # equivalent to l1_weight in DETR
    cfg.MODEL.POET.CTRS = 0.5 # l2 center keypoint coefficient
    cfg.MODEL.POET.DELTAS = 0.5 # l1 offsets coefficient
    cfg.MODEL.POET.KPTS_CLASS = 0.2 # Visibility coefficient

    # TRANSFORMER
    cfg.MODEL.POET.NHEADS = 8
    cfg.MODEL.POET.DROPOUT = 0.1
    cfg.MODEL.POET.DIM_FEEDFORWARD = 2048
    cfg.MODEL.POET.ENC_LAYERS = 6
    cfg.MODEL.POET.DEC_LAYERS = 6
    cfg.MODEL.POET.PRE_NORM = False

    cfg.MODEL.POET.HIDDEN_DIM = 256
    cfg.MODEL.POET.NUM_OBJECT_QUERIES = 100

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1 #MAA: Potentially change this?
