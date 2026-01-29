from typing import List

import numpy as np

# from ..ort_runner import ORTRunner, FinalResult
from ..ort_runner import softmax


from ..trt_runner import TRTRunner, FinalResult


class SE2020(TRTRunner):
    def __init__(self, config:dict):
        super().__init__(config)

    def postProcess(self, outputs_blob:List[np.ndarray]) -> FinalResult:
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
                print(f" - {self.config['classes_name'][i]}", end=' ')
        print()

        # 6. 格式转化
        ## 注意 SE2020 默认是 0 表示“变化”，1 表示“未变化”，这里需要转换为 0 表示“未变化”，255 表示“变化”
        mask_bin = (~mask_bin[0]).astype(np.uint8)*255 # (1, 512, 512) -> (512, 512)
        
        output1 = output1[0].astype(np.uint8) # (1, 512, 512) -> (512, 512)
        output2 = output2[0].astype(np.uint8) # (1, 512, 512) -> (512, 512)
        
        return FinalResult(mask_bin=mask_bin, mask_1=output1, mask_2=output2)