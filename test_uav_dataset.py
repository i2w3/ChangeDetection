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
from src import UAV_Dataset, logger


plt.style.use("dark_background")

def parse_args():
    parser = argparse.ArgumentParser(description="在 UAV2 数据集上进行测试")
    parser.add_argument("model", type=str, help="使用的模型")
    parser.add_argument("--dataset_path", "-dp", type=str, default="./res/data/UAV2",
                        help="UAV2 数据集路径")
    parser.add_argument("--sample_name", "-sn", type=str, default="im1",
                        help="数据集路径下子样本名称，默认读取 ./res/data/UAV2/im1* 文件")
    parser.add_argument("--over_lap", "-ol", type=int, default=None,
                        help="是否在裁切图片时加入 over_lap，以消除边缘效应，None 则表示全图预测，over_lap 为 0 表示不重叠裁切，大于 0 则表示重叠裁切")
    parser.add_argument("--zip_times", "-zt", type=int, default=1,
                        help="对输入图像进行压缩的次数，每次压缩为原来的一半，默认压缩一次")
    parser.add_argument("--enable_more_test", "-emt", action="store_true",
                        help="是否进行更多的测试")
    parser.add_argument("--config_path", type=str, default="./config.json",
                        help="配置文件路径")
    return parser.parse_args()


def plot_data(config:dict, 
              img_data:np.ndarray,
              mask_data:np.ndarray, 
              fig:Optional[plt.Figure] = None, 
              save_str:Optional[str] = None) -> plt.Figure:
    '''绘制图像数据和预测结果
    '''
    if fig is None:
        fig = plt.figure(figsize=(15, 10))
    fig.clf() # 清除之前的内容
    axes = fig.subplots(2, 2)
    ## Row 1
    axes[0,0].imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("img")
    axes[0,0].axis('off')
    legend_elements = []
    for i, class_name in enumerate(config["classes_name"]):
        color = np.array(config["classes_cmap"][i]) / 255.0
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label=class_name, markerfacecolor=color, markersize=10))
    axes[0,1].axis('off')
    axes[0,1].legend(handles=legend_elements, loc='center', fontsize='large')
    ## Row 2
    mask, img = over_leap(config, img_data, mask_data)
    axes[1,0].imshow(img)
    axes[1,0].set_title("mask_img")
    axes[1,0].axis('off')
    axes[1,1].imshow(mask, cmap='gray')
    axes[1,1].set_title("mask_bin")
    axes[1,1].axis('off')

    plt.tight_layout()
    if save_str is not None:
        plt.savefig(save_str)
    return fig


def over_leap(config:dict, img_data:np.ndarray, mask_data:np.ndarray) -> List[np.ndarray]:
    '''图像叠加
    '''
    img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != mask_data.shape[:2]:
        mask_data = cv2.resize(mask_data, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 将 mask 不为 0 的区域，使用 config["classes_cmap"] 中对应的颜色进行替换
    color_mask = np.zeros_like(img)
    for i, color in enumerate(config["classes_cmap"]):
        if i == 0:
            color_mask[mask_data == i] = img[mask_data == i]
        else:
            color_mask[mask_data == i] = color
    # 将 mask_data 不为 0 的区域，使用 255 替换
    mask_data_bin = np.where(mask_data == 0, 0, 255).astype(np.uint8)
    return [mask_data_bin,color_mask]


if __name__ == "__main__":
    args = parse_args()
    logger('info', f"Arguments: {args}")
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        logger('info', f"Loaded config from {args.config_path}: {config[args.model]}")

    model = getattr(src, args.model)(config[args.model])
    dataset = UAV_Dataset(args.dataset_path)
    sample = dataset.sub_gen(args.sample_name)
    # 压缩 sample 分辨率，否则目标太大
    print(f"Sample shape: {sample.shape}")
    for _ in range(args.zip_times):
        sample = cv2.resize(sample, (sample.shape[1] // 2, sample.shape[0] // 2))
    print(f"{'ZIP ' * args.zip_times}Sample shape: {sample.shape}")
    pred = model(sample, over_lap=args.over_lap)

    fig = plot_data(config[args.model], sample, pred, 
                    save_str=f"images/SS_{args.model}-{Path(args.dataset_path).stem}-{args.sample_name}-{'zt' + str(args.zip_times)}-{str(args.over_lap)}.png")
    # plt.show()

    if not args.enable_more_test:
        raise SystemExit("Only run single test!")
    
    # 进行更多测试
    TIME_LIST = []
    LAND_TYPE = [x.stem for x in dataset.data_root.iterdir() if x.is_file()]
    fig = plt.figure(figsize=(15, 10))
    for i in LAND_TYPE:
        sample = dataset.sub_gen(i)
        start_time = time.time()
        print(f"Sample shape: {sample.shape}")
        for _ in range(args.zip_times):
            sample = cv2.resize(sample, (sample.shape[1] // 2, sample.shape[0] // 2))
        print(f"{'ZIP ' * args.zip_times}Sample shape: {sample.shape}")
        pred = model(sample, over_lap=args.over_lap)
        end_time = time.time()
        TIME_LIST.append(end_time - start_time)   
        plot_data(config[args.model], sample, pred, fig=fig, 
                  save_str=f"images/SS_{args.model}-{Path(args.dataset_path).stem}-{i}-{'zt' + str(args.zip_times)}-{str(args.over_lap)}.png")
    plt.close('all')
    logger('info', f"Average processing time: {np.mean(TIME_LIST):.4f} seconds")