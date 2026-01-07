import argparse
import gc
import json
import time
from typing import List, Union, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import *


def parse_args():
    parser = argparse.ArgumentParser(description="在 GVLM_CD 数据集上进行测试")
    parser.add_argument("--dataset_path", "-dp", type=str, default="./res/data/GVLM_CD",
                        help="GVLM_CD 数据集路径")
    parser.add_argument("--sample_name", "-sn", type=str, default="A Luoi_Vietnam",
                        help="数据集路径下子样本名称，默认读取 ./res/data/GVLM_CD/A Luoi_Vietnam 文件夹下的数据")
    parser.add_argument("--cut_size", "-cs", type=int, default=None,
                        help="裁剪子图的大小，默认不裁剪使用全图")
    parser.add_argument("--enable_force_grouth", "-efg", action="store_true",
                        help="是否通过后处理强迫模型仅关注地面变化")
    parser.add_argument("--enable_more_test", "-emt", action="store_true",
                        help="是否使用相同的随机种子在 GVLM_CD 数据集 17 种地形上均裁剪子图进行预测")
    parser.add_argument("--run_time", type=int, default=100,
                        help="每种地形裁剪子图的次数")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，None 表示不固定，仅供单次测试使用")
    parser.add_argument("--config_path", type=str, default="./config.json",
                        help="配置文件路径")
    return parser.parse_args()


def plot_data(args:argparse.Namespace, config:dict, img_data:GVLM_Sample, mask_data:SE2020OUTPUT, fig:Optional[plt.Figure] = None, save_str:Optional[str] = None) -> plt.Figure:
    '''绘制图像数据和预测结果
    '''
    if fig is None:
        fig = plt.figure(figsize=(15, 10))
    fig.clf()
    axes = fig.subplots(2, 3)
    ## Row 1
    axes[0,0].imshow(cv2.cvtColor(img_data.img_A, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("img_A")
    axes[0,0].axis('off')
    axes[0,1].imshow(cv2.cvtColor(img_data.img_B, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title("img_B")
    axes[0,1].axis('off')
    axes[0,2].imshow(img_data.img_ref, cmap='gray')
    axes[0,2].set_title("ref")
    axes[0,2].axis('off')
    ## Row 2
    axes[1,0].imshow(over_leap(img_data.img_A, mask_data.mask_1))
    axes[1,0].set_title("mask_A")
    axes[1,0].axis('off')
    axes[1,1].imshow(over_leap(img_data.img_B, mask_data.mask_2))
    axes[1,1].set_title("mask_B")
    axes[1,1].axis('off')
    if args.enable_force_grouth:
        axes[1,2].imshow(mask_data.mask_bin, cmap='gray')
        axes[1,2].set_title("mask_bin")
        axes[1,2].axis('off')
    else:
        # plot lengends for all classes
        legend_elements = []
        for i, class_name in enumerate(config["SE2020"]["classes_name"]):
            color = np.array(config["SE2020"]["classes_cmap"][i]) / 255.0
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label=class_name,
                                              markerfacecolor=color, markersize=10))
        axes[1,2].axis('off')
        axes[1,2].legend(handles=legend_elements, loc='center', fontsize='large')

    plt.tight_layout()
    if save_str is not None:
        plt.savefig(save_str)
    return fig


def calc_mIoU(pred_mask:np.ndarray, true_mask:np.ndarray, cls_list:List[int]) -> float:
    '''计算 mIoU
    '''
    if pred_mask.shape != true_mask.shape:
        print("Warning: Resizing true_mask to match pred_mask shape for mIoU calculation.")
        true_mask = cv2.resize(true_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
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


def over_leap(img:np.ndarray, img_mask:Union[Image.Image, np.ndarray], alpha:float=0.5) -> np.ndarray:
    '''图像叠加
    '''
    if isinstance(img_mask, Image.Image):
        img_mask = np.array(img_mask.convert("RGB"))
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)
    elif isinstance(img_mask, np.ndarray):
        img_mask = Image.fromarray(img_mask).convert("P")
        img_mask.putpalette(model.color_map)
        img_mask = np.array(img_mask.convert("RGB"))
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)
    else:
        raise TypeError("img_mask must be PIL.Image or np.ndarray")
    if img.shape[:2] != img_mask.shape[:2]:
        img_mask = cv2.resize(img_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(img, alpha, img_mask, 1 - alpha, 0)


if __name__ == "__main__":
    args = parse_args()
    logger('info', "Testing GVLM_CD dataset with SE2020 model, warning: 预训练模型与数据集不匹配，仅作测试使用!")
    logger('info', f"Arguments: {args}")
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        logger('info', f"Loaded config from {args.config_path}: {config}")

    model = SE2020(config["SE2020"])
    dataset = GVLM_CDataset(args.dataset_path)
    sample = dataset.sub_gen(args.sample_name, args.cut_size, args.seed) # 随机裁剪1024x1024大小的图像块
    # sample_enhance, enchance_imgB = UAV_enchance(sample, 512)

    pred = model(sample.img_A, sample.img_B)
    if args.enable_force_grouth:
        pred.mask_1[pred.mask_1 != 2] = 0 # 强迫仅关注地面变化类别(id 2)
        pred.mask_2[pred.mask_2 != 2] = 0
        # 在 SE2020 中的 mask_bin，1 表示未变化，0 表示有变化，但是不包含类别信息，所以需要根据 mask_1 和 mask_2 重新计算地面变化
        pred.mask_bin = pred.mask_1 + pred.mask_2
        pred.mask_bin[pred.mask_bin != 0] = 255
        logger('info', f"mIoU: {calc_mIoU(pred.mask_bin, sample.img_ref, [0,255]):.4f}") # mIoU 计算仅支持强迫关注地面变化的情况
    else:
        pred.mask_1 = Image.fromarray(pred.mask_1).convert("P")
        pred.mask_1.putpalette(model.color_map)

        pred.mask_2 = Image.fromarray(pred.mask_2).convert("P")
        pred.mask_2.putpalette(model.color_map)

    fig = plot_data(args, config, sample, pred, save_str="images/gvlm_result.png")
    # plt.show()
    plt.close('all')
    
    if not args.enable_more_test:
        raise SystemExit("Only run single test!")
    
    # 批量测试
    MIOU_LIST = []
    TIME_LIST = []
    LAND_TYPE = [x.name for x in dataset.data_root.iterdir() if x.is_dir()]
    fig = plt.figure(figsize=(15, 10))
    for i in LAND_TYPE:
        for j in range(args.run_time):
            sample = dataset.sub_gen(i, args.cut_size, j)
            # sample_enhance, enchance_imgB = UAV_enchance(sample, 512)
            start_time = time.time()
            pred = model(sample.img_A, sample.img_B)
            if args.enable_force_grouth:
                pred.mask_1[pred.mask_1 != 2] = 0 # 强迫仅关注地面变化类别(id 2)
                pred.mask_2[pred.mask_2 != 2] = 0
                # 在 SE2020 中的 mask_bin，1 表示未变化，0 表示有变化，但是不包含类别信息，所以需要根据 mask_1 和 mask_2 重新计算地面变化
                pred.mask_bin = pred.mask_1 + pred.mask_2
                pred.mask_bin[pred.mask_bin != 0] = 255
                mIOU = calc_mIoU(pred.mask_bin, sample.img_ref, [0,255])
                MIOU_LIST.append(mIOU)
            else:
                pred.mask_1 = Image.fromarray(pred.mask_1).convert("P")
                pred.mask_1.putpalette(model.color_map)

                pred.mask_2 = Image.fromarray(pred.mask_2).convert("P")
                pred.mask_2.putpalette(model.color_map)
            end_time = time.time()
            TIME_LIST.append(end_time - start_time)
            
            plot_data(args, config, sample, pred, fig=fig, save_str=f"images/{i}-{j}.png")
            del sample, pred
            if j % 10 == 0:
                gc.collect()
    plt.close('all')
    if np.sum(MIOU_LIST) > 0:
        logger('info', f"Average mIoU: {np.mean(MIOU_LIST):.4f}")
    logger('info', f"Average processing time: {np.mean(TIME_LIST):.4f} seconds")