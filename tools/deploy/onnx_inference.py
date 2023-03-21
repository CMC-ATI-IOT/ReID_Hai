# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import sys
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
sys.path.append('../..')
sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup
from fastreid.modeling import build_model
import cv2
import numpy as np
import onnxruntime
import tqdm
from demo.predictor import FeatureExtractionDemo
from torch.backends import cudnn

cudnn.benchmark = True

def get_parser():
    parser = argparse.ArgumentParser(description="onnx model inference")

    parser.add_argument(
        "--model-path",
        default="/home/thangnv/fast-reid/output/onnx_model/ma_cu1_2_3_pku_mars_duk_msmt_ep60_batch256.onnx",
        help="onnx model path"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='onnx_output',
        help='path to save converted caffe model'
    )
    parser.add_argument(
        "--height",
        type=int,
        default=384,
        help="height of image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="width of image"
    )
    return parser


def preprocess(image_path, image_height, image_width):
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img


def normalize(nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)



def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("/home/thangnv/fast-reid/configs/Market1501/sbs_S50_onnx.yml")    
    cfg.freeze()
    # default_setup(cfg)
    return cfg
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features
if __name__ == "__main__":

    args = get_parser().parse_args()

    ort_sess = onnxruntime.InferenceSession(args.model_path)

    input_name = ort_sess.get_inputs()[0].name

    if not os.path.exists(args.output): os.makedirs(args.output)
    
    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            image = preprocess(path, args.height, args.width)
            feat = ort_sess.run(None, {input_name: image})[0]
            print("before norm: ")
            print(feat)
            feat_ = normalize(feat, axis=1)
            print("after norm: ")
            print(feat_)            
            np.save(os.path.join(args.output, path.replace('.jpg', '.npy').split('/')[-1]), feat)

    cfg1 = setup_cfg()
    demo = FeatureExtractionDemo(cfg1)
    img = cv2.imread("/home/thangnv/fast-reid/demo/mct/1/192.168.1.6_2_13.jpg")
    feat1 = demo.run_on_image(img)
    print("feat1 pth before norm ",feat1)
    feat1_ = postprocess(feat1)
    print("feat1 pth after norm ",feat1_)

    result = cosine_similarity(feat,feat1)
    print("cosine similarity before norm:", result)

    result = cosine_similarity(feat_,feat1_)
    print("cosine similarity before norm:", result)

    np.testing.assert_allclose(feat_, feat1_, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")