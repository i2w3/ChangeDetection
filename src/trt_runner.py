from dataclasses import dataclass
from typing import List
from pathlib import Path
import time

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from .logger import logger
from .utils import FinalResult


class TRTRunner:
    def __init__(self, config:dict):
        self.config = config
        self.color_map = [c for sublist in self.config["classes_cmap"] for c in sublist]  # [num_classes, 3] -> [num_classes * 3]

        # setting trt environment
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.config["engine_path"], "rb") as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            logger.error(f"Failed to load TensorRT engine from {self.config['engine_path']}")
            raise RuntimeError("TensorRT engine deserialization failed.")
        
        # create ExecutionContext & CUDA Stream
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # IO containers initialization
        self.bindings = [] # device pointers list
        self.inputs = {}
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i) # input, seg_out, aux_out
            mode = self.engine.get_tensor_mode(name) # INPUT, OUTPUT
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))  # TRT to numpy.ndarray dtype
            shape = list(self.engine.get_tensor_shape(name))
            self.bindings.append(None)  # placeholder
            info = {"name": name,
                    "dtype": dtype,
                    "device": None,
                    "shape": shape,
                    "dummy": np.zeros(shape, dtype=dtype),
                    "index": i}
            if mode == trt.TensorIOMode.INPUT:
                self.inputs[name] = info
            else:
                self.outputs[name] = info
        logger("info", "model input data:")
        for input_data in self.inputs.values():
            logger("info", f"\t{input_data['name']} - {input_data['dtype']}{input_data['shape']}")
            self.context.set_input_shape(input_data['name'], input_data['shape'])
        logger("info", "model outputs data:")
        for output_name in self.outputs.values():
            logger("info", f"\t{output_name['name']} - {output_name['shape']}")

        # warm up
        start_time = time.time()
        for info in list(self.inputs.values()) + list(self.outputs.values()):
            vol = 1
            for s in info["shape"]:
                vol *= s
            size = int(vol * np.dtype(info["dtype"]).itemsize)
            device_mem = cuda.mem_alloc(size)
            info["device"] = device_mem
            self.bindings[info["index"]] = int(device_mem)

        for name, info in self.inputs.items():
            cuda.memcpy_htod_async(info["device"], info["dummy"], self.stream)
        self.stream.synchronize()
        self.context.execute_v2(self.bindings)
        
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
        # swapRB and scale to [0,1]
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0/255.0,size=input_size,mean=(0,0,0),swapRB=True,ddepth=cv2.CV_32F)
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
        
        raise NotImplementedError("infer method need to be implemented in subclass.")