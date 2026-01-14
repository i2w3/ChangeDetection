from typing import List

import numpy as np

from ..ort_runner import ORTRunner, FinalResult
from ..ort_runner import sigmoid, logger, cv2, time


class DEFO(ORTRunner):
    def __init__(self, config:dict):
        super().__init__(config)

    def preProcess(self, img:np.ndarray, use_one:bool = True) -> np.ndarray:
        shape = img.shape[:2]
        input_size = (self.config["model_size"], self.config["model_size"])
        if shape[0] != self.config["model_size"] or shape[1] != self.config["model_size"]:
            img = cv2.resize(img, input_size)
        if use_one:
            mean_array = np.array(self.config["model_mean"], np.float32).reshape(1,3,1,1)
            std_array  = np.array(self.config["model_std"] , np.float32).reshape(1,3,1,1)
        else:
            mean_array = np.array(self.config["model_mean2"], np.float32).reshape(1,3,1,1)
            std_array  = np.array(self.config["model_std2"] , np.float32).reshape(1,3,1,1)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0,size=input_size,mean=(0,0,0),swapRB=True,ddepth=cv2.CV_32F) # only swapRB and scale
        blob = (blob - mean_array) / std_array
        return blob

    def postProcess(self, outputs_blob:List[np.ndarray]) -> FinalResult:
        output_bin, output1, output2,  = outputs_blob
        output_bin = sigmoid(output_bin)
        if output_bin.ndim == 4:
            output_bin = output_bin.squeeze(1)
        mask_bin = (output_bin > 0.5)
        mask_bin = mask_bin.astype(np.uint8)

        np_pa = np.argmax(output1, axis=1).astype(np.uint8) * mask_bin
        np_pb = np.argmax(output2, axis=1).astype(np.uint8) * mask_bin
        
        return FinalResult(mask_bin=mask_bin[0], mask_1=np_pa[0], mask_2=np_pb[0])
    
    def infer(self, image1: np.ndarray, image2: np.ndarray):
        start_time = time.time()
        input1_blob = self.preProcess(image1)
        input2_blob = self.preProcess(image2, use_one=False)
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