from typing import List

import numpy as np

from ..ort_runner import ORTRunner, FinalResult
from ..ort_runner import sigmoid


class SCanNet(ORTRunner):
    def __init__(self, config:dict):
        super().__init__(config)

    def postProcess(self, outputs_blob:List[np.ndarray]) -> FinalResult:
        output_bin, output1, output2,  = outputs_blob
        output_bin = sigmoid(output_bin)
        if output_bin.ndim == 4:
            output_bin = output_bin.squeeze(1)
        mask_bin = (output_bin > 0.5)
        mask_bin = mask_bin.astype(np.uint8)

        np_pa = np.argmax(output1, axis=1).astype(np.uint8) * mask_bin
        np_pb = np.argmax(output2, axis=1).astype(np.uint8) * mask_bin

        # 反转 np_cm, 对齐可视化效果
        mask_bin[mask_bin==0] = 255
        mask_bin[mask_bin==1] = 0
        
        return FinalResult(mask_bin=mask_bin[0], mask_1=np_pa[0], mask_2=np_pb[0])