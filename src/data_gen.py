from dataclasses import dataclass
from typing import Tuple, Union, Optional
from pathlib import Path

import cv2
import numpy as np


def random_center(h, w, model_size) -> Tuple[int, int]:
    '''随机生成中心点坐标
    '''
    xc = np.random.randint(model_size, w - model_size)
    yc = np.random.randint(model_size, h - model_size)
    return xc, yc


def glob_img(img_path:Path, img_name:str) -> Optional[Path]:
    '''根据文件名模糊匹配图像文件
    '''
    try:
        img_file = list(img_path.glob(f"{img_name}.*"))[0]
        return img_file
    except IndexError:
        print(f"Cannot find image file in {img_path.resolve()}/{img_name}.*")
        return None


@dataclass
class GVLM_Sample:
    img_A: np.ndarray
    img_B: np.ndarray
    img_ref: np.ndarray
    img_id: str


class GVLM_CDataset:
    def __init__(self, data_root:Union[Path, str]):
        if not isinstance(data_root, Path):
            data_root = Path(data_root)
        self.data_root = data_root


    def sub_gen(self, sub_folder:Union[Path, str], cut_size:Optional[int]=None, seed:Optional[int]=None) -> GVLM_Sample:
        '''读取数据集的子文件夹，随机裁剪数据
        '''
        if not isinstance(sub_folder, Path):
            sub_folder = Path(sub_folder)

        if cut_size is None:
            half_size = None
        else:
            half_size = cut_size // 2  # 半尺寸
        
        img_root = self.data_root / sub_folder
        if not img_root.exists():
            raise FileNotFoundError(f"Path {img_root.resolve()} not found.")
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed()

        img_A_path   = glob_img(img_root, "im1")
        img_B_path   = glob_img(img_root, "im2")
        img_ref_path = glob_img(img_root, "ref")

        img_A    = cv2.imread(str(img_A_path))
        img_B    = cv2.imread(str(img_B_path))
        img_ref  = cv2.imread(str(img_ref_path), cv2.IMREAD_GRAYSCALE)
        if img_A is None or img_B is None:
            raise ValueError(f"Failed to read images in {img_root.resolve()}.")
        if img_ref is None:
            img_ref = np.zeros(img_A.shape[:2], dtype=np.uint8)

        # 判断是否需要裁剪
        if half_size is None:
            return GVLM_Sample(
                img_A   = img_A,
                img_B   = img_B,
                img_ref = img_ref,
                img_id  = sub_folder.name
            )
        else:
            xc, yc = random_center(*img_A.shape[:2], half_size)
            return GVLM_Sample(
                img_A   = img_A[yc - half_size: yc + half_size, xc - half_size: xc + half_size],
                img_B   = img_B[yc - half_size: yc + half_size, xc - half_size: xc + half_size],
                img_ref = img_ref[yc - half_size: yc + half_size, xc - half_size: xc + half_size],
                img_id  = sub_folder.name + "_{0}_{1}".format(xc, yc)
            )
    