from typing import List

import numpy as np

from ..ort_runner import ORTRunner, FinalResult
from ..ort_runner import sigmoid


class ClearSCD(ORTRunner):
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

        return FinalResult(mask_bin=mask_bin[0], mask_1=np_pa[0], mask_2=np_pb[0])