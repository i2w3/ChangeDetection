import argparse
import gc
import json
import time
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

import src
from src import GVLM_CDataset, GVLM_Sample, FinalResult, logger


def parse_args():
    parser = argparse.ArgumentParser(description="在 GVLM_CD 数据集上进行测试")
    parser.add_argument("model", type=str, help="使用的模型")
    parser.add_argument("--dataset_path", "-dp", type=str, default="./res/data/GVLM_CD",
                        help="GVLM_CD 数据集路径")
    parser.add_argument("--sample_name", "-sn", type=str, default="A Luoi_Vietnam",
                        help="数据集路径下子样本名称，默认读取 ./res/data/GVLM_CD/A Luoi_Vietnam 文件夹下的数据")
    parser.add_argument("--cut_size", "-cs", type=int, default=None,
                        help="裁剪子图的大小，默认不裁剪使用全图")
    parser.add_argument("--enable_more_test", "-emt", action="store_true",
                        help="是否使用相同的随机种子在 GVLM_CD 数据集 17 种地形上均裁剪子图进行预测")
    parser.add_argument("--run_time", "-rt", type=int, default=100,
                        help="每种地形裁剪子图的次数")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，None 表示不固定，仅供单次测试使用")
    parser.add_argument("--config_path", type=str, default="./config.json",
                        help="配置文件路径")
    return parser.parse_args()


def plot_data(config:dict, 
              img_data:GVLM_Sample,
              mask_data:FinalResult, 
              fig:Optional[plt.Figure] = None, 
              save_str:Optional[str] = None) -> plt.Figure:
    '''绘制图像数据和预测结果
    '''
    if fig is None:
        fig = plt.figure(figsize=(15, 10))
    fig.clf() # 清除之前的内容
    axes = fig.subplots(2, 3)
    ## Row 1
    axes[0,0].imshow(cv2.cvtColor(img_data.img_A, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("img_A")
    axes[0,0].axis('off')
    axes[0,1].imshow(cv2.cvtColor(img_data.img_B, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title("img_B")
    axes[0,1].axis('off')
    legend_elements = []
    for i, class_name in enumerate(config["classes_name"]):
        color = np.array(config["classes_cmap"][i]) / 255.0
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label=class_name, markerfacecolor=color, markersize=10))
    axes[0,2].axis('off')
    axes[0,2].legend(handles=legend_elements, loc='center', fontsize='large')
    ## Row 2
    img_A, img_B, mask = over_leap(config, img_data, mask_data)
    axes[1,0].imshow(img_A)
    axes[1,0].set_title("mask_A")
    axes[1,0].axis('off')
    axes[1,1].imshow(img_B)
    axes[1,1].set_title("mask_B")
    axes[1,1].axis('off')
    axes[1,2].imshow(mask, cmap='gray')
    axes[1,2].set_title("mask_bin")
    axes[1,2].axis('off')

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


def over_leap(config:dict, img_data:GVLM_Sample, mask_data:FinalResult) -> List[np.ndarray]:
    '''图像叠加
    '''
    img_A, img_B = img_data.img_A, img_data.img_B
    img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
    mask_bin, mask_1, mask_2 = mask_data.mask_bin, mask_data.mask_1, mask_data.mask_2
    if img_A.shape[:2] != mask_1.shape[:2]:
        mask_bin = cv2.resize(mask_bin, (img_A.shape[1], img_A.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_1 = cv2.resize(mask_1, (img_A.shape[1], img_A.shape[0]), interpolation=cv2.INTER_NEAREST)
        # mask_2 = cv2.resize(mask_2, (img_A.shape[1], img_A.shape[0]), interpolation=cv2.INTER_NEAREST) #TODO: shape img_B != img_A?
    if img_B.shape[:2] != mask_2.shape[:2]:
        mask_2 = cv2.resize(mask_2, (img_B.shape[1], img_B.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 将 mask_1 不为 0 的区域，使用 config["classes_cmap"] 中对应的颜色进行替换
    color_mask_1 = np.zeros_like(img_A)
    color_mask_2 = np.zeros_like(img_B)
    for i, color in enumerate(config["classes_cmap"]):
        if i == 0:
            color_mask_1[mask_1 == i] = img_A[mask_1 == i]
            color_mask_2[mask_2 == i] = img_B[mask_2 == i]
        else:
            color_mask_1[mask_1 == i] = color
            color_mask_2[mask_2 == i] = color
    return color_mask_1, color_mask_2, mask_bin


if __name__ == "__main__":
    args = parse_args()
    logger('info', "Testing GVLM_CD dataset with SE2020 model, warning: 预训练模型与数据集不匹配，仅作测试使用!")
    logger('info', f"Arguments: {args}")
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        logger('info', f"Loaded config from {args.config_path}: {config[args.model]}")

    model = getattr(src, args.model)(config[args.model])
    dataset = GVLM_CDataset(args.dataset_path)
    sample = dataset.sub_gen(args.sample_name, args.cut_size, args.seed)
    pred = model(sample.img_A, sample.img_B)

    fig = plot_data(config[args.model], sample, pred, save_str=f"images/{args.model}-{Path(args.dataset_path).stem}-{args.sample_name}.png")
    plt.show()

    if not args.enable_more_test:
        raise SystemExit("Only run single test!")
    
    # 进行更多测试
    TIME_LIST = []
    LAND_TYPE = [x.name for x in dataset.data_root.iterdir() if x.is_dir()]
    fig = plt.figure(figsize=(15, 10))
    for i in LAND_TYPE:
        SAVE_PATH = Path("images") / Path(i)
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        for j in range(args.run_time):
            sample = dataset.sub_gen(i, args.cut_size, j)
            start_time = time.time()
            pred = model(sample.img_A, sample.img_B)
            
            end_time = time.time()
            TIME_LIST.append(end_time - start_time)
            
            plot_data(config[args.model], sample, pred, fig=fig, save_str=str(SAVE_PATH / f"{i}-{j}.png"))
            del sample, pred
            if j % 10 == 0:
                gc.collect()
    plt.close('all')
    logger('info', f"Average processing time: {np.mean(TIME_LIST):.4f} seconds")