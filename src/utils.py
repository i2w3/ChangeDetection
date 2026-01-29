from dataclasses import dataclass

import numpy as np


@dataclass
class FinalResult:
    mask_bin: np.ndarray
    mask_1: np.ndarray
    mask_2: np.ndarray