from dataclasses import dataclass
from typing import List

import numpy as np

from .ort_runner import ORTRunner, softmax


@dataclass
class SE2020OUTPUT:
    '''SE2020 的训练数据集为前后时相的两张图片各自对应一张标注图，表示发生变化的区域以及该图片变化区域内各时期的土地性质，所以有两个 mask 图输出
    '''
    mask_bin: np.ndarray
    mask_1: np.ndarray
    mask_2: np.ndarray


class SE2020(ORTRunner):
    def __init__(self, config:dict):
        super().__init__(config)

    def postProcess(self, outputs_blob:List[np.ndarray]) -> SE2020OUTPUT:
        output1, output2, output_bin = outputs_blob
        # 1. 分割分支 Softmax
        output1 = softmax(output1, axis=1)
        output2 = softmax(output2, axis=1)

        # 2. 获取类型索引
        output1 = np.argmax(output1, axis=1) + 1
        output2 = np.argmax(output2, axis=1) + 1
        
        # 3. 处理二值分支 (output_bin > 0.5 表示“未变化”)
        if output_bin.ndim == 4:
            output_bin = output_bin.squeeze(1)
        mask_bin = (output_bin > 0.5)

        # 4. 应用二值掩码
        output1[mask_bin] = 0
        output2[mask_bin] = 0

        # 5. 打印变化类别
        print("Changed classes in image2:",end=' ')
        for i in np.unique(output2).tolist():
            if i != 0:
                print(f" - {self.classes_name[i]}", end=' ')
        print()

        # 6. 格式转化
        mask_bin = mask_bin[0].astype(np.uint8) * 255 # (1, 512, 512) -> (512, 512)
        output1 = output1[0].astype(np.uint8) # (1, 512, 512) -> (512, 512)
        output2 = output2[0].astype(np.uint8) # (1, 512, 512) -> (512, 512)
        
        return SE2020OUTPUT(mask_bin=mask_bin, mask_1=output1, mask_2=output2)