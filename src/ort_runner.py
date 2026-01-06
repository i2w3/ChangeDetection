from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort


def softmax(x, axis=1):
    '''Softmax
    '''
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class ORTRunner:
    def __init__(self, config:dict):
        model_path:str = config["model_path"] # ONNX 模型路径
        model_size:int = config["model_size"] # 模型输入尺寸
        self.model_path = Path(model_path)
        self.model_size = model_size
        self.model_mean:list[float] = config["model_mean"]
        self.model_std:list[float]  = config["model_std"]
        assert len(self.model_mean) == 3, "model_mean length must be 3!"
        assert len(self.model_std) == 3, "model_std length must be 3!"
        self.model_mean_bgr255:list[int] = (self.model_mean[2]*255, 
                                            self.model_mean[1]*255, 
                                            self.model_mean[0]*255)
        self.model_std_array:np.ndarray = np.array(self.model_std).reshape(1, 3, 1, 1).astype(np.float32)
        self.classes_name:list[str] = config["classes_name"]
        self.classes_cmap:list[list[int]] = config["classes_cmap"]
        assert len(self.classes_name) == len(self.classes_cmap), "classes_name and classes_cmap length mismatch!"
        self.color_map = [c for sublist in self.classes_cmap for c in sublist]  # [num_classes, 3] -> [num_classes * 3]
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
                'trt_engine_cache_path': f'{self.model_path.parent}/trt_cache',
                'trt_timing_cache_enable': True, # timing cache 加速在其它设备上建立 engine
                'trt_timing_cache_path': f'{self.model_path.parent}/trt_cache/time_cache',
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
        self.session = ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)
        end_time = time.time()
        load_time_ms = (end_time - start_time) * 1000
        print(f"Model loaded in {load_time_ms:.2f} ms.")
        # print name
        self.inputs_with_dummy = {}
        self.outputs = []
        print("model input name:")
        for input_name in self.session.get_inputs():
            print(f"\t{input_name.name} - {input_name.shape}")
            self.inputs_with_dummy[input_name.name] = np.zeros(input_name.shape, dtype=np.float32)
        print("model outputs name:")
        for output_name in self.session.get_outputs():
            print(f"\t{output_name.name} - {output_name.shape}")
            self.outputs.append(output_name.name)
        # warm up
        start_time = time.time()
        self.session.run(None, self.inputs_with_dummy)
        end_time = time.time()
        load_time_ms = (end_time - start_time) * 1000
        print(f"Model warm up in {load_time_ms:.2f} ms.")

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)
    
    def preProcess(self, src:np.ndarray) -> np.ndarray:
        '''预处理
        '''
        img = src.copy()
        shape = src.shape[:2]  # [height, width, channel] -> [height, width]
        if (shape[0] != self.model_size or shape[1] != self.model_size):
            img = cv2.resize(img, (self.model_size, self.model_size))
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (self.model_size, self.model_size), self.model_mean_bgr255, swapRB=True, crop=False, ddepth=cv2.CV_32F)
        blob /= self.model_std_array
        return blob
    
    def postProcess(self, *args, **kwargs):
        raise NotImplementedError("postProcess method need to be implemented in subclass.")
    
    def infer(self, image1: np.ndarray, image2: np.ndarray, run_time:int=1):
        preProcess_time  = []
        inference_time   = []
        postProcess_time = []
        for _ in range(run_time):
            start_time = time.time()
            input1_blob = self.preProcess(image1)
            input2_blob = self.preProcess(image2)
            preProcess_time.append(time.time() - start_time)

            start_time = time.time()
            outputs_blob:list = self.session.run(self.outputs, {"input1": input1_blob, "input2": input2_blob})
            inference_time.append(time.time() - start_time)
            
            start_time = time.time()
            result = self.postProcess(outputs_blob)
            postProcess_time.append(time.time() - start_time)
        print(f"Average Pre-Process Time:  {np.mean(preProcess_time)*1000:.2f} ms")
        print(f"Average Inference Time:   {np.mean(inference_time)*1000:.2f} ms")
        print(f"Average Post-Process Time: {np.mean(postProcess_time)*1000:.2f} ms")
        return result