import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src import GVLM_CDataset, Matcher


def parse_args():
    parser = argparse.ArgumentParser(description="TODO")
    parser.add_argument("method", type=str, help="使用的方法")
    parser.add_argument("--dataset_path", "-dp", type=str, default="./res/data/UAV",
                        help="UAV 数据集路径")
    parser.add_argument("--sample_name", "-sn", type=str, default="demo4",
                        help="数据集路径下子样本名称，默认读取 ./res/data/UAV/demo4 文件夹下的数据")
    parser.add_argument("--enable_cuda", "-cuda", action="store_true",
                        help="是否启用 cuda 加速")
    parser.add_argument("--num_features", "-nf", type=int, default=10000,
                        help="特征点数量")
    parser.add_argument("--num_gmatches", "-gm", type=int, default=10,
                        help="好的匹配最少需要的点数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = GVLM_CDataset(args.dataset_path)
    sample = dataset.sub_gen(args.sample_name)

    start_time = time.time()
    matcher = Matcher(args.method, args.num_features, args.num_gmatches, args.enable_cuda)
    ov1, ov2 = matcher(sample.img_A, sample.img_B)
    end_time = time.time()
    print(f"计算重叠区域耗时: {end_time - start_time:.4f} 秒")

    if ov1 is not None and ov2 is not None:
        mask_A, mask_B = Matcher.get_mask(sample.img_A, sample.img_B, ov1, ov2)
        r_A = Matcher.get_rotate_crop_image(sample.img_A, ov1)
        r_B = Matcher.get_rotate_crop_image(sample.img_B, ov2)
        # cv2.imwrite("overlap_mask_A.png", r_A)
        # cv2.imwrite("overlap_mask_B.png", r_B)
        
        fig = plt.figure(figsize=(15, 10))
        axes = fig.subplots(2, 3)
        axes[0, 0].imshow(cv2.cvtColor(sample.img_A, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("img_A")
        axes[0, 0].axis('off')
        axes[1, 0].imshow(cv2.cvtColor(sample.img_B, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("img_B")
        axes[1, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(cv2.polylines(sample.img_A.copy(), [np.int32(ov1)], True, (0, 0, 255), 3), cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("img_A with overlap zone")
        axes[0, 1].axis('off')
        axes[1, 1].imshow(cv2.cvtColor(cv2.polylines(sample.img_B.copy(), [np.int32(ov2)], True, (0, 0, 255), 3), cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("img_B with overlap zone")
        axes[1, 1].axis('off')

        axes[0, 2].imshow(cv2.cvtColor(r_A, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("img_A rotated crop")
        axes[0, 2].axis('off')
        axes[1, 2].imshow(cv2.cvtColor(r_B, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("img_B rotated crop")
        axes[1, 2].axis('off')

        fig.suptitle(f"Matcher: {matcher.method_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"./images/match_{matcher.method_name}.png", dpi=300)
        plt.show()