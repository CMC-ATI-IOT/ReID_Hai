
# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import argparse
import io
import sys
import numpy as np 

import onnx
import onnx.optimizer as onnxoptimizer
import torch
from onnxsim import simplify
from torch.onnx import OperatorExportTypes
import onnxruntime

sys.path.append('.')

import fastreid
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.file_io import PathManager
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.logger import setup_logger

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("/home/thangnv/fast-reid/configs/Market1501/sbs_S50_onnx.yml")
    
    cfg.freeze()
    return cfg

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':

    
    cfg = setup_cfg()
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    if cfg.MODEL.HEADS.POOL_LAYER == 'FastGlobalAvgPool':
        cfg.MODEL.HEADS.POOL_LAYER = 'GlobalAvgPool'
    model = build_model(cfg)
    Checkpointer(model).load("/home/thangnv/fast-reid/logs/ma_cu1_2_3_pku_mars_duk_msmt_ep60_batch256/model_final.pth")
    if hasattr(model.backbone, 'deploy'):
        model.backbone.deploy(True)
    model.eval()
    input = torch.randn(1, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]).to(model.device)
    raw_output = model(input)


    ort_sess = onnxruntime.InferenceSession("/home/thangnv/fast-reid/outputs/onnx_model/ma_cu1_2_3_pku_mars_duk_msmt_ep60_batch256.onnx")

    input_name = ort_sess.get_inputs()[0].name
    ort_input = {ort_sess.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_sess.run(None, ort_input)	

    print("compare ONNX Runtime and PyTorch results") 
    np.testing.assert_allclose(to_numpy(raw_output), ort_outs[0], rtol=1e-03, atol=1e-04)
    print(raw_output.shape)
    print(ort_outs[0].shape)
    for i in range(1024): 
        if float(raw_output[0][i].cpu()) != float(ort_outs[0][0][i]):
            print(i, float(raw_output[0][i].cpu()) , float(ort_outs[0][0][i]))
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
