from typing import List, Optional

import numpy as np

from ..ort_runner import ORTRunner
from ..ort_runner import softmax, logger, cv2, time


class LoveDA(ORTRunner):
    def __init__(self, config:dict):
        super().__init__(config)

    def preProcess(self, img:np.ndarray) -> np.ndarray:
        shape = img.shape[:2]
        input_size = (self.config["model_size"], self.config["model_size"])
        if shape[0] != self.config["model_size"] or shape[1] != self.config["model_size"]:
            img = cv2.resize(img, input_size)
        mean_array = np.array(self.config["model_mean"], np.float32).reshape(1,3,1,1)
        std_array  = np.array(self.config["model_std"] , np.float32).reshape(1,3,1,1)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0,size=input_size,mean=(0,0,0),swapRB=True,ddepth=cv2.CV_32F) # only swapRB
        blob = (blob - mean_array) / std_array
        return blob

    def postProcess(self, outputs_blob:List[np.ndarray]) -> np.ndarray:
        output_bin = outputs_blob[0]
        np_bin = np.argmax(output_bin, axis=1).astype(np.uint8)
        return np_bin[0]
        
    
    def infer(self, image: np.ndarray, over_lap:Optional[int] = None) -> np.ndarray:
        start_time = time.time()
        if over_lap is not None:
            h, w = image.shape[:2] # 图像高宽
            model_size = self.config["model_size"] # 切片大小
            stride = model_size - over_lap # 步长 = 切片大小 - 重叠大小
            full_probs = np.zeros((len(self.config["classes_name"]), h, w), dtype=np.float32) # 存储最终的概率图   
            count_map = np.zeros((1, h, w), dtype=np.float32) # 计数矩阵，记录每个像素被预测叠加了多少次

            # 生成切片坐标：高度方向和宽度方向
            h_steps = []
            c_h = 0
            while c_h + model_size <= h:
                h_steps.append(c_h)
                c_h += stride
            if len(h_steps) == 0 or h_steps[-1] != h - model_size:
                h_steps.append(max(0, h - model_size))
            w_steps = []
            c_w = 0
            while c_w + model_size <= w:
                w_steps.append(c_w)
                c_w += stride
            if len(w_steps) == 0 or w_steps[-1] != w - model_size:
                w_steps.append(max(0, w - model_size))
            # 对每个切片进行推理，并将概率图累加到 full_probs 中
            for idx_h, h_s in enumerate(h_steps):
                h_e = h_s + model_size
                for idx_w, w_s in enumerate(w_steps):
                    w_e = w_s + model_size
                    img = image[h_s: h_e, w_s: w_e, :]
                    # cv2.imwrite(f"./logs/temp_{idx_h}_{idx_w}.png", img)
                    input_blob = self.preProcess(img)
                    outputs_blob:list = self.session.run(self.outputs, {"input": input_blob})

                    # 获取 logits
                    logits = outputs_blob[0] # list[np.ndarray] -> C, H, W
                    logits = softmax(logits)
                    full_probs[:, h_s: h_e, w_s: w_e] += logits[0]
                    count_map[:, h_s: h_e, w_s: w_e] += 1
            # 最终通过 argmax 获取类别索引
            full_probs /= np.maximum(count_map, 1.0)
            output = np.argmax(full_probs, axis=0).astype(np.uint8)
        else:
            input_blob = self.preProcess(image)
            outputs_blob:list = self.session.run(self.outputs, {"input": input_blob})
            output = self.postProcess(outputs_blob)
        end_time = time.time()
        logger('info', f"Inference time: {end_time - start_time:.3f} seconds")
        return output
    

class EarthVQA(LoveDA):
    def __init__(self, config:dict):
        super().__init__(config)