# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import tqdm
import onnxruntime


TRT_LOGGER = trt.Logger()


def get_parser():
    parser = argparse.ArgumentParser(description="trt model inference")

    parser.add_argument(
        "--model-path",
        default="outputs/trt_model/baseline.engine",
        help="trt model path"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="trt_output",
        help="path to save trt model inference results"
    )
    parser.add_argument(
        '--batch-size',
        default=1,
        type=int,
        help='the maximum batch size of trt module'
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="height of image"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="width of image"
    )
    return parser


class HostDeviceMem(object):
    """ Host and Device Memory Package """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

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

class TrtEngine:

    def __init__(self, trt_file=None, gpu_idx=0, batch_size=1):
        cuda.init()
        self._batch_size = batch_size
        self._device_ctx = cuda.Device(gpu_idx).make_context()
        self._engine = self._load_engine(trt_file)
        self._context = self._engine.create_execution_context()
        self._input, self._output, self._bindings, self._stream = self._allocate_buffers(self._context)

    def _load_engine(self, trt_file):
        """
        Load tensorrt engine.
        :param trt_file:    tensorrt file.
        :return:
            ICudaEngine
        """
        with open(trt_file, "rb") as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _allocate_buffers(self, context):
        """
        Allocate device memory space for data.
        :param context:
        :return:
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self._engine:
            size = trt.volume(self._engine.get_binding_shape(binding)) * self._engine.max_batch_size
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self._engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def infer(self, data):
        """
        Real inference process.
        :param model:   Model objects
        :param data:    Preprocessed data
        :return:
            output
        """
        # Copy data to input memory buffer
        [np.copyto(_inp.host, data.ravel()) for _inp in self._input]
        # Push to device
        self._device_ctx.push()
        # Transfer input data to the GPU.
        # cuda.memcpy_htod_async(self._input.device, self._input.host, self._stream)
        [cuda.memcpy_htod_async(inp.device, inp.host, self._stream) for inp in self._input]
        # Run inference.
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)
        # Transfer predictions back from the GPU.
        # cuda.memcpy_dtoh_async(self._output.host, self._output.device, self._stream)
        [cuda.memcpy_dtoh_async(out.host, out.device, self._stream) for out in self._output]
        # Synchronize the stream
        self._stream.synchronize()
        # Pop the device
        self._device_ctx.pop()

        return [out.host.reshape(self._batch_size, -1) for out in self._output[::-1]]

    def inference_on_images(self, imgs, new_size=(384, 128)):
        trt_inputs = []
        for img in imgs:
            input_ndarray = self.preprocess(img, *new_size)
            trt_inputs.append(input_ndarray)
        trt_inputs = np.vstack(trt_inputs)

        valid_bsz = trt_inputs.shape[0]
        if valid_bsz < self._batch_size:
            trt_inputs = np.vstack([trt_inputs, np.zeros((self._batch_size - valid_bsz, 3, *new_size))])

        result, = self.infer(trt_inputs)
        result = result[:valid_bsz]
        feat = self.postprocess(result, axis=1)
        return feat

    @classmethod
    def preprocess(cls, img, img_height, img_width):
        # Apply pre-processing to image.
        resize_img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        type_img = resize_img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
        return type_img

    @classmethod
    def postprocess(cls, nparray, order=2, axis=-1):
        """Normalize a N-D numpy array along the specified axis."""
        norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
        return nparray / (norm + np.finfo(np.float32).eps)

    def __del__(self):
        del self._input
        del self._output
        del self._stream
        self._device_ctx.detach()  # release device context


if __name__ == "__main__":
    args = get_parser().parse_args()

    trt = TrtEngine(args.model_path, batch_size=args.batch_size)

    if not os.path.exists(args.output): os.makedirs(args.output)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for img_path in tqdm.tqdm(args.input):
            img = cv2.imread(img_path)
            # the model expects RGB inputs
            cvt_img = img[:, :, ::-1]
            feat = trt.inference_on_images([cvt_img])
            print(feat)


    path = "/home/thangnv/fast-reid/demo/mct/1/192.168.1.6_2_13.jpg"
    ort_sess = onnxruntime.InferenceSession("/home/thangnv/fast-reid/output/onnx_model/ma_cu1_2_3_pku_mars_duk_msmt_ep60_batch256.onnx")

    input_name = ort_sess.get_inputs()[0].name
    # output_name = ort_sess.get_outputs()[0].name

    # print(output_name)
    image = preprocess(path, args.height, args.width)
    feat1 = ort_sess.run(None, {input_name: image})[0]
    # print("before norm: ")
    # print(feat1)
    feat_ = normalize(feat1, axis=1)
    print("onnx output: ")
    print(feat_)            
    np.testing.assert_allclose(feat, feat_, rtol=1e-3, atol=1e-3)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")