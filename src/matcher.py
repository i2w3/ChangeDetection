from typing import Tuple, Optional, List

import cv2
import numpy as np

from .logger import logger


class Matcher:
    def __init__(self, method:str, num_features:int, num_gmatches:int, enable_cuda:bool = False):
        self.method = method.upper()
        self.num_features = num_features
        self.num_gmatches = num_gmatches
        self.enable_cuda = enable_cuda
        self.detector = None
        self.bf = None
        if self.enable_cuda:
            if not cv2.cuda.getCudaEnabledDeviceCount():
                raise RuntimeError("OPENCV[CUDA] ERROR!")
            logger("info", "ENABLE CUDA ACCELERATION")
            self.enable_cuda = True
            if self.method == "ORB":
                self.detector = cv2.cuda.ORB_create(self.num_features, blurForDescriptor = True)
                self.bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        else:
            if self.method == "SIFT":
                self.detector = cv2.SIFT_create(self.num_features)
                self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
            elif self.method == "SURF":
                logger("warning", "SURF is patented, make sure you have the right to use it!")
                self.detector = cv2.xfeatures2d_SURF.create(self.num_features)
                self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
            elif self.method == "ORB":
                self.detector = cv2.ORB_create(self.num_features)
                self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)
            elif self.method == "AKAZE":
                self.detector = cv2.AKAZE_create()
                logger("warning", "AKAZE method not allow modift num of features!")
                self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)
        self.method_name = f"{self.method}{'-cuda' if self.enable_cuda else ''}"
        assert self.detector is not None and self.bf is not None, f"{self.method_name} not support!"
    
    def __call__(self, img1:np.ndarray, img2:np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # 计算单应性矩阵 H(3*3)
        if self.enable_cuda:
            H = self._gpu(img1, img2)
        else:
            H = self._cpu(img1, img2)
        if H is None:
            logger("error", "Calculate Homography Matrix Failed!")
            return None, None
        
        # 计算重叠区域多边形
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        # A. 定义 Img1 的矩形轮廓 (在 Img1 坐标系)
        img1_rect = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        # B. 定义 Img2 的矩形轮廓 (在 Img2 坐标系)
        img2_rect = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        # C. 将 Img2 轮廓投影到 Img1 坐标系
        img2_rect_transformed = cv2.perspectiveTransform(img2_rect, H)
        # D. 计算交集 (intersectConvexConvex)
        # 输入必须是凸多边形（矩形和透视变换后的矩形都是凸的）
        # 返回值：ret (面积), intersection_points (交集点)
        ret, overlap_poly_on_img1 = cv2.intersectConvexConvex(img1_rect, img2_rect_transformed)

        if overlap_poly_on_img1 is None or ret <= 0:
            logger("warning", "No Overlap Area Found!")
            return None, None
        # 此时 overlap_poly_on_img1 是重叠区域在 Img1 上的坐标，需要将其映射回 Img2，得到 overlap_poly_on_img2

        # E. 计算 H 的逆矩阵
        H_inv = np.linalg.inv(H)

        # F. 将重叠多边形由 Img1 坐标系 -> Img2 坐标系
        overlap_poly_on_img2 = cv2.perspectiveTransform(overlap_poly_on_img1, H_inv)

        # 返回两个多边形坐标 (N, 1, 2) float32
        return overlap_poly_on_img1, overlap_poly_on_img2


    def _cpu(self, img1:np.ndarray, img2:np.ndarray) -> Optional[np.ndarray]:
        ''' 计算单应性矩阵 H (特征提取 + 特征匹配 + 计算 H)
        '''
        # 1. 预处理：灰度化
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 2. 特征提取
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        # 3. 特征匹配
        good_matches = self.feature_matching(des1, des2)
        if good_matches is None:
            return None

        # 4. 计算单应性矩阵 H (Img2 -> Img1)
        return self.get_homography(good_matches, kp1, kp2)

    
    def _gpu(self, img1:np.ndarray, img2:np.ndarray) -> Optional[np.ndarray]:
        ''' 计算单应性矩阵 H (特征提取 + 特征匹配 + 计算 H)
        '''
        # 1. 预处理：灰度化
        gpu_img1 = cv2.cuda_GpuMat()
        gpu_img2 = cv2.cuda_GpuMat()
        gpu_img1.upload(img1)
        gpu_img2.upload(img2)
        gray1 = cv2.cuda.cvtColor(gpu_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cuda.cvtColor(gpu_img2, cv2.COLOR_BGR2GRAY)
        
        # 2. 特征提取
        gpu_kp1, des1 = self.detector.detectAndComputeAsync(gray1, None)
        gpu_kp2, des2 = self.detector.detectAndComputeAsync(gray2, None)

        # 3. 特征匹配
        good_matches = self.feature_matching(des1, des2)
        if good_matches is None:
            return None

        # 4. 计算单应性矩阵 H (Img2 -> Img1)
        kp1 = self.detector.convert(gpu_kp1)
        kp2 = self.detector.convert(gpu_kp2)
        return self.get_homography(good_matches, kp1, kp2)
    
    def feature_matching(self, des1:np.ndarray, des2:np.ndarray) -> Optional[List[cv2.DMatch]]:
        ''' Lowe's ratio test 筛选特征进行匹配
        '''
        raw_matches = self.bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in raw_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        if len(good_matches) < self.num_gmatches:
            logger("warning", f"Match points is not enough ({len(good_matches)} < {self.num_gmatches})!")
            return None
        return good_matches
    
    def get_homography(self, good_matches:List[cv2.DMatch], kp1:List[cv2.KeyPoint], kp2:List[cv2.KeyPoint]) -> Optional[np.ndarray]:
        ''' 计算单应性矩阵 H (Img2 -> Img1)
        '''
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    @staticmethod
    def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
        # 1. 计算最小外接矩形 (Rotated Rectangle)
        # 注意：cv2.minAreaRect 可以直接接收多边形点集
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # 2. 对矩形的四个点进行排序 (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        # 这一步是为了防止透视变换后图像颠倒或扭曲
        pts = box.reshape(4, 2)
        rect_std = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect_std[0] = pts[np.argmin(s)]             # TL
        rect_std[2] = pts[np.argmax(s)]             # BR

        diff = np.diff(pts, axis=1)
        rect_std[1] = pts[np.argmin(diff)]          # TR
        rect_std[3] = pts[np.argmax(diff)]          # BL

        # 3. 计算变换后的宽高
        # width: TL-TR 和 BL-BR 的最大距离
        width_A = np.linalg.norm(rect_std[0] - rect_std[1])
        width_B = np.linalg.norm(rect_std[2] - rect_std[3])
        img_crop_width = int(max(width_A, width_B))

        # height: TL-BL 和 TR-BR 的最大距离
        height_A = np.linalg.norm(rect_std[0] - rect_std[3])
        height_B = np.linalg.norm(rect_std[1] - rect_std[2])
        img_crop_height = int(max(height_A, height_B))

        # 4. 定义目标坐标 (标准矩形)
        dst_pts = np.array([
            [0, 0],
            [img_crop_width - 1, 0],
            [img_crop_width - 1, img_crop_height - 1],
            [0, img_crop_height - 1]
        ], dtype="float32")

        # 5. 获取透视变换矩阵 M
        M = cv2.getPerspectiveTransform(rect_std, dst_pts)

        # 6. 变换图像 (Warp Image)
        # 此时得到的是包含了重叠区域外接矩形的图像
        dst_img = cv2.warpPerspective(
            img, 
            M, 
            (img_crop_width, img_crop_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, # 边界填充黑色
            borderValue=(0, 0, 0)
        )

        # 7. 核心逻辑：变换多边形并进行遮盖 (Masking)
        # 将原始多边形 points 投影到新的坐标系下
        # 输入需要是 float32 类型
        points_float = points.astype(np.float32)
        if points_float.ndim == 2: # 确保是 (N, 1, 2) 或 (N, 2)
            points_float = points_float.reshape(-1, 1, 2)
            
        # 使用同一个矩阵 M 变换多边形顶点
        transformed_poly = cv2.perspectiveTransform(points_float, M)
        
        # 创建一个与变换后图像大小一致的 Mask
        mask = np.zeros((img_crop_height, img_crop_width), dtype=np.uint8)
        
        # 在 Mask 上绘制变换后的多边形 (白色)
        cv2.fillPoly(mask, [np.int32(transformed_poly)], 255)
        
        # 对图像应用 Mask (保留白色区域，其他变黑)
        dst_img = cv2.bitwise_and(dst_img, dst_img, mask=mask)

        # 8. (可选) 保持你原有的旋转逻辑，如果长宽比不合适则旋转90度
        if img_crop_height * 1.0 / img_crop_width >= 1.5:
            dst_img = np.rot90(dst_img)

        return dst_img

    @staticmethod
    def get_mask(img_A, img_B, ov1, ov2, inverse=False):
        # 1. 创建全黑的单通道 Mask
        # 注意：为了进行位运算，Mask 的值通常设为 255 (白色) 而不是 1
        mask_A = np.zeros(img_A.shape[:2], dtype=np.uint8)
        mask_B = np.zeros(img_B.shape[:2], dtype=np.uint8)

        # 2. 填充多边形区域
        if ov1 is not None:
            cv2.fillPoly(mask_A, [np.int32(ov1)], 255)
        if ov2 is not None:
            cv2.fillPoly(mask_B, [np.int32(ov2)], 255)

        # 3. (可选) 如果你想反过来：把重叠区域遮住，显示背景
        if inverse:
            mask_A = cv2.bitwise_not(mask_A)
            mask_B = cv2.bitwise_not(mask_B)

        # 4. 应用 Mask 到原始图像
        # bitwise_and 会将 mask 中为 0 的位置在原图中也置为 0，为 255 的位置保留原值
        masked_img_A = cv2.bitwise_and(img_A, img_A, mask=mask_A)
        masked_img_B = cv2.bitwise_and(img_B, img_B, mask=mask_B)

        return masked_img_A, masked_img_B