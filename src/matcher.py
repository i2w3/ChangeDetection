import cv2
import numpy as np


from .data_gen import GVLM_Sample

def UAV_enchance(gvlm_sample:GVLM_Sample, model_size:int) -> tuple[GVLM_Sample | None, np.ndarray | None]:
    '''针对 GVLM_CD 数据集进行增强，模拟 UAV 视角下的图像变化，注意仅裁剪 img_A 和 img_ref 图像块，并变化 img_B 的视角，但是 img_B 不进行裁剪，用来做图像匹配算法研究
    '''
    # 1. 找出标签中的主要变化区域，主要针对此区域进行增强
    # cv2.RETR_EXTERNAL: 只找外轮廓（通常找最大物体时不需要内部孔洞的轮廓）
    # cv2.CHAIN_APPROX_SIMPLE: 压缩轮廓点，节省内存
    contours, hierarchy = cv2.findContours(gvlm_sample.img_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea).squeeze() # (num_points, 1, 2) -> (num_points, 2)
        center = max_contour.mean(axis=0).astype(int)
        h, w = gvlm_sample.img_ref.shape[:2]
        half_size = model_size // 2
        if center[0] - half_size < 0:
            center[0] = half_size
        if center[0] + half_size > w:
            center[0] = w - half_size
        if center[1] - half_size < 0:
            center[1] = half_size
        if center[1] + half_size > h:
            center[1] = h - half_size
        gvlm_sample_enhanced = GVLM_Sample(
            img_A   = gvlm_sample.img_A[center[1] - half_size: center[1] + half_size, center[0] - half_size: center[0] + half_size],
            img_B   = gvlm_sample.img_B[center[1] - half_size: center[1] + half_size, center[0] - half_size: center[0] + half_size],
            img_ref = gvlm_sample.img_ref[center[1] - half_size: center[1] + half_size, center[0] - half_size: center[0] + half_size],
            img_id  = f"{gvlm_sample.img_id}_{center[0]}_{center[1]}_{model_size}" 
        )
        return gvlm_sample_enhanced, random_fly(gvlm_sample.img_B)
    else:
        # 找不到最大轮廓，图片无地形变化，直接返回
        return gvlm_sample, None
    

def random_fly(src:np.ndarray) -> np.ndarray:
    '''模拟第二次 UAV 飞行时，拍摄的图片与第一次飞行无法匹配
    '''
    # 定义变换参数：轻微旋转(angle)、缩放(scale)、平移(tx, ty)
    angle = 2.0  # 旋转 2 度
    scale = 1.05 # 放大 5% (模拟高度降低)
    tx, ty = 50, -50 # 平移 (x, y) 像素
    rows, cols = src.shape[:2]
    H = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
    H[0, 2] += tx
    H[1, 2] += ty
    return cv2.warpAffine(src, H, (cols, rows), borderMode=cv2.BORDER_REFLECT)