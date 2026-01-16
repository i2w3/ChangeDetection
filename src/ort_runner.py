from dataclasses import dataclass
from typing import List
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from .logger import logger


def softmax(x:np.ndarray, axis:int=1):
    '''Softmax
    '''
    assert x.ndim == 4, "only support 4-D tensor"
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def sigmoid(x:np.ndarray):
    '''Sigmoid
    '''
    return 1 / (1 + np.exp(-x))


@dataclass
class FinalResult:
    mask_bin: np.ndarray
    mask_1: np.ndarray
    mask_2: np.ndarray


class ORTRunner:
    def __init__(self, config:dict):
        self.config = config
        self.color_map = [c for sublist in self.config["classes_cmap"] for c in sublist]  # [num_classes, 3] -> [num_classes * 3]
        # setting ort environment
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # 启用所有优化
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 4 * 1024 * 1024 * 1024, # 4 GB
                'trt_int8_enable': False,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': f'{Path(self.config["model_path"]).parent}/trt_cache',
                'trt_timing_cache_enable': True, # timing cache 加速在其它设备上建立 engine
                'trt_timing_cache_path': f'{Path(self.config["model_path"]).parent}/trt_cache/time_cache',
                'trt_force_timing_cache': False, # 仅在与生成 timing cache 的 GPU 型号完全相同的 GPU 上使用
            }),
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            ('CPUExecutionProvider', {})
        ]
        # load ONNX model, and try to convert to TensorRT engine
        start_time = time.time()
        self.session = ort.InferenceSession(self.config["model_path"], sess_options=so, providers=providers)
        end_time = time.time()
        load_time_ms = (end_time - start_time) * 1000
        logger("info", f"Model loaded in {load_time_ms:.2f} ms.")
        # print name
        self.inputs_with_dummy = {}
        self.outputs = []
        logger("info", "model input name:")
        for input_name in self.session.get_inputs():
            logger("info", f"\t{input_name.name} - {input_name.shape}")
            self.inputs_with_dummy[input_name.name] = np.zeros(input_name.shape, dtype=np.float32)
        logger("info", "model outputs name:")
        for output_name in self.session.get_outputs():
            logger("info", f"\t{output_name.name} - {output_name.shape}")
            self.outputs.append(output_name.name)
        # warm up
        start_time = time.time()
        self.session.run(None, self.inputs_with_dummy)
        end_time = time.time()
        load_time_ms = (end_time - start_time) * 1000
        logger("info", f"Model warm up in {load_time_ms:.2f} ms.")

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)
    
    def preProcess(self, img:np.ndarray) -> np.ndarray:
        shape = img.shape[:2]
        input_size = (self.config["model_size"], self.config["model_size"])
        if shape[0] != self.config["model_size"] or shape[1] != self.config["model_size"]:
            img = cv2.resize(img, input_size)
        mean_array = np.array(self.config["model_mean"], np.float32).reshape(1,3,1,1)
        std_array  = np.array(self.config["model_std"] , np.float32).reshape(1,3,1,1)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0/255.0,size=input_size,mean=(0,0,0),swapRB=True,ddepth=cv2.CV_32F) # only swapRB and scale
        blob = (blob - mean_array) / std_array
        return blob
    
    def postProcess(self, outputs_blob:List[np.ndarray]) -> FinalResult:
        '''后处理
        '''
        raise NotImplementedError("postProcess method need to be implemented in subclass.")
    
    def infer(self, image1: np.ndarray, image2: np.ndarray):
        start_time = time.time()
        input1_blob = self.preProcess(image1)
        input2_blob = self.preProcess(image2)
        preProcess_time = time.time() - start_time

        start_time = time.time()
        outputs_blob:list = self.session.run(self.outputs, {"input1": input1_blob, "input2": input2_blob})
        inference_time = time.time() - start_time
        
        start_time = time.time()
        result = self.postProcess(outputs_blob)
        postProcess_time = time.time() - start_time
        
        logger("info", f"Average Pre-Process Time:  {preProcess_time*1000:.2f} ms")
        logger("info", f"Average Inference Time:   {inference_time*1000:.2f} ms")
        logger("info", f"Average Post-Process Time: {postProcess_time*1000:.2f} ms")
        return result