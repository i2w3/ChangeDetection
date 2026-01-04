import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src import *


def calc_mIOU(pred_mask:np.ndarray, true_mask:np.ndarray, cls_list:list[int]) -> float:
    iou_list = []
    for cls in cls_list:
        pred_cls = (pred_mask == cls).astype(int)
        true_cls = (true_mask == cls).astype(int)

        intersection = np.sum(pred_cls * true_cls)
        union = np.sum(pred_cls) + np.sum(true_cls) - intersection

        if union == 0:
            iou = 1.0  # 如果该类别在预测和真实中都不存在，视为完全匹配
        else:
            iou = intersection / union

        iou_list.append(iou)

    mIOU = np.mean(iou_list)
    return mIOU


if __name__ == "__main__":
    print("Testing GVLM_CD dataset with SE2020 model, warning: 预训练模型与数据集不匹配，仅作测试使用!")
    force_grouth = True # SE2020 模型可以关注多种类型变化，通过后处理强迫仅关注地面变化
    seed = None  # 随机种子，None 表示不固定
    config_path = "./config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    model = SE2020(config["SE2020"])
    dataset = GVLM_CDataset("./res/data/GVLM_CD")
    sample = dataset.sub_gen("A Luoi_Vietnam", 1024, seed) # 随机裁剪1024x1024大小的图像块
    sample_enhance, enchance_imgB = UAV_enchance(sample, 512)

    pred = model(sample_enhance.img_A, sample_enhance.img_B)
    if force_grouth:
        pred.mask_1[pred.mask_1 != 2] = 0 # 强迫仅关注地面变化类别(id 2)
        pred.mask_2[pred.mask_2 != 2] = 0
        # 在 SE2020 中的 mask_bin，1 表示未变化，0 表示有变化，但是不包含类别信息，所以需要根据 mask_1 和 mask_2 重新计算地面变化
        pred.mask_bin = pred.mask_1 + pred.mask_2
        pred.mask_bin[pred.mask_bin != 0] = 255
        print(f"mIOU: {calc_mIOU(pred.mask_bin, sample_enhance.img_ref, [0,255]):.4f}") # mIOU 计算仅支持强迫关注地面变化的情况

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ## Row 1
    axes[0,0].imshow(cv2.cvtColor(sample_enhance.img_A, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("img_A")
    axes[0,0].axis('off')
    axes[0,1].imshow(cv2.cvtColor(sample_enhance.img_B, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title("img_B")
    axes[0,1].axis('off')
    axes[0,2].imshow(sample_enhance.img_ref, cmap='gray')
    axes[0,2].set_title("ref")
    axes[0,2].axis('off')
    ## Row 2
    axes[1,0].imshow(pred.mask_1)
    axes[1,0].set_title("mask_A")
    axes[1,0].axis('off')
    axes[1,1].imshow(pred.mask_2)
    axes[1,1].set_title("mask_B")
    axes[1,1].axis('off')
    axes[1,2].imshow(pred.mask_bin, cmap='gray')
    axes[1,2].set_title("mask_bin")
    axes[1,2].axis('off')

    plt.tight_layout()
    # plt.savefig(f"result_{sample.img_id}.png")
    plt.show()