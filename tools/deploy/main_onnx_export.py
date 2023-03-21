"""
@author: sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

import torch
sys.path.append('../..')
sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup
from fastreid.modeling import build_model
#from fastreid.export.tensorflow_export import export_tf_reid_model
import onnx
from torch.backends import cudnn
import torch.nn.functional as F
import io
from fastreid.utils.checkpoint import Checkpointer
import numpy as np
import onnxruntime
import cv2
cudnn.benchmark = True
def setup():

    cfg = get_cfg()
    cfg.merge_from_file("/home/thangnv/fast-reid/configs/Market1501/sbs_S50_onnx.yml")    
    cfg.freeze()
    # default_setup(cfg)
    return cfg
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
if __name__ == "__main__":
    cfg1 = setup()
    cfg = cfg1.clone()
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = build_model(cfg)
    #model.load_state_dict( torch.load('logs/Vechile1M/bagtricks_R50-ibn_4gpu/model_0211199.pth'))
    Checkpointer(model).load('/home/thangnv/fast-reid/logs/ma_cu1_2_3_pku_mars_duk_msmt_ep60_batch256/model_final.pth')

    model.cuda()
    model.eval()
    #print (model)
    layer_names = {}
    # for name, layer in model.named_modules():
    #     layer_names[layer] = name
    # print("torch ops name:", layer_names)
    dummy_inputs = torch.randn(1, 3, 384, 128).cuda()
    
    # torch.onnx.export(model,  # model being run
    #                 {'images': dummy_inputs},  # model input (or a tuple for multiple inputs)
    #                 "reid_test.onnx",  # where to save the model (can be a file or file-like object)
    # export_params=True,  # store the trained parameter weights inside the model file
    # )
    # onnx_model = onnx.load("reid_test.onnx")
    # onnx.checker.check_model(onnx_model)
    original_image = cv2.imread("/home/thangnv/fast-reid/inputs/test.png")

    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]
    # Apply pre-processing to image.
    image = cv2.resize(original_image, (128, 384), interpolation=cv2.INTER_CUBIC)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
    img = image.cuda()

    with torch.no_grad():
        torch_out1 = model.forward({'images': dummy_inputs})
        torch_out = F.normalize(torch_out1)

    ort_session = onnxruntime.InferenceSession("/home/thangnv/fast-reid/outputs/onnx_model/ma_cu1_2_3_pku_mars_duk_msmt_ep60_batch256.onnx")

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_inputs)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]
    norm = np.linalg.norm(img_out_y, ord=2, axis=1, keepdims=True)
    print ('norm:', norm)
    feat = img_out_y/norm
    np.savetxt('th_re.txt', to_numpy(torch_out).T)
    np.savetxt('onnx_re.txt', feat.T)  
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), feat, rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")