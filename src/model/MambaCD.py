from typing import List

import numpy as np

from ..ort_runner import ORTRunner, FinalResult


class MambaCD(ORTRunner):
    def __init__(self, config:dict):
        super().__init__(config)

    def postProcess(self, outputs_blob:List[np.ndarray]) -> FinalResult:
        output_bin, output1, output2,  = outputs_blob

        np_cm = np.argmax(output_bin, axis=1).squeeze().astype(np.uint8)
        np_pa = np.argmax(output1, axis=1).astype(np.uint8) * np_cm
        np_pb = np.argmax(output2, axis=1).astype(np.uint8) * np_cm
                
        return FinalResult(mask_bin=np_cm, mask_1=np_pa[0], mask_2=np_pb[0])