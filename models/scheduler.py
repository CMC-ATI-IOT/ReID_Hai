# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# Based on: https://github.com/facebookresearch/detectron2/blob/master/detectron2/solver/build.py

import sys, os
sys.path.append('.')
lib_path = os.path.abspath(os.path.join('models'))
sys.path.append(lib_path)

import copy
import itertools
import math
import re
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

import torch

from fastreid.config import CfgNode
from fastreid.utils.params import ContiguousParams
import lr_scheduler


def build_lr_scheduler(cfg, optimizer, iters_per_epoch):
    max_epoch = cfg.SOLVER.MAX_EPOCH - max(
        math.ceil(cfg.SOLVER.WARMUP_ITERS / iters_per_epoch), cfg.SOLVER.DELAY_EPOCHS)

    scheduler_dict = {}

    scheduler_args = {
        "MultiStepLR": {
            "optimizer": optimizer,
            # multi-step lr scheduler options
            "milestones": cfg.SOLVER.STEPS,
            "gamma": cfg.SOLVER.GAMMA,
        },
        "CosineAnnealingLR": {
            "optimizer": optimizer,
            # cosine annealing lr scheduler options
            "T_max": max_epoch,
            "eta_min": cfg.SOLVER.ETA_MIN_LR,
        },

    }

    scheduler_dict["lr_sched"] = getattr(lr_scheduler, cfg.SOLVER.SCHED)(
        **scheduler_args[cfg.SOLVER.SCHED])

    if cfg.SOLVER.WARMUP_ITERS > 0:
        warmup_args = {
            "optimizer": optimizer,

            # warmup options
            "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
            "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
            "warmup_method": cfg.SOLVER.WARMUP_METHOD,
        }
        scheduler_dict["warmup_sched"] = lr_scheduler.WarmupLR(**warmup_args)

    return scheduler_dict
