# file tree
```bash
+---
|   config.json # 所有模型的配置文件
|   test_gvlm_dataset.py # 测试模型在 GVLM 数据集上的表现
|   test_matcher.py # 测试图像匹配算法
|   test_time.py # 解析 log，获取模型平均运行时间
+---images # 存放一些展示图
+---res # 存放数据集和模型路径
|   \---data
|       +---GVLM_CD # GVLM_CD 数据集格式
|       |   \---<LAND_TYPE>
|       |           im1.png
|       |           im2.png
|       |           ref.png
|       +---SECOND # 按照 GVLM 数据集存放格式
|       \---UAV
+---scripts # 一些模型转换脚本
\---src
    |   data_gen.py # 读数数据集
    |   logger.py # 单例模式的 logger
    |   matcher.py # 图像匹配算法
    |   ort_runner.py # onnxruntime
    |   trt_runner.py # tensorRT
    |   
    \---model # 继承 ort_runner.py 的模型运行代码
```

# model list
all onnx model can download from [huggingface](https://huggingface.co/i2w3/model_zoo)

## [SE2020](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection)
![](./images/SE2020.png)

## [MambaCD](https://github.com/ChenHongruixuan/MambaCD)
此模型由于有自定义 cuda 算子，转换为 onnx 后与原始 pt 模型输出存在明显差异

- 原始 pt 格式的运行效果
![](./images/MambaCD_pt.png)

- onnx 转换后的效果
![](./images/MambaCD_onnx.png)
```bash
!only for linux, windows no cu129 and cu130 build pytorch
mamba create -n cu129 python=3.12 pytorch=2.8.0=cuda129_mkl_py312_* torchvision=0.24.0=cuda129* torchaudio=2.8.0=*cuda129* cuda-toolkit cuda-version=12.9 triton=3.4.0 timm=0.4.12 -c nvidia
mamba activate cu129
git clone https://github.com/ChenHongruixuan/MambaCD.git
cd MambaCD
pip install -r requirements.txt
cd kernels/selective_scan && pip install . --no-build-isolation
mamba install mmengine mmcv mmdet mmsegmentation ftfy regex
pip install opencv-python-headless mmpretrain
```

## [SCanNet](https://github.com/DingLei14/SCanNet)
SCanNet 的官方代码中，img_A 和 img_B 使用不同的 mean/std 参数，但是实测差别不大
- 普通RGB标准化参数
![](./images/SCanNet.png)

- 官方双参数
![](./images/SCanNet2.png)

```bash
git clone https://github.com/DingLei14/SCanNet
cd SCanNet
```
```python
import torch

from models.SCanNet import SCanNet as Net

model = Net(3, 7).cuda()
model.load_state_dict(torch.load("models/SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth"))
model = model.cuda()
model.eval()

dummy_input1 = torch.rand(1, 3, 512, 512).cuda()
dummy_input2 = torch.rand(1, 3, 512, 512).cuda()

with torch.no_grad():
    output_bcd, output_T1, output_T2 = model(dummy_input1, dummy_input2)

print(output_bcd.shape)
print(output_T1.shape)
print(output_T2.shape)

onnx_program = torch.onnx.export(
    model, 
    (dummy_input1, dummy_input2), 
    "./SCanNet_SECOND.onnx",
    export_params=True,
    input_names=['input1', 'input2'],
    output_names=['output_bin', 'output1', 'output2'],
    opset_version=17,
    external_data=False,
    verbose=False
)
```

# more model
[open-cd](https://github.com/likyoo/open-cd) base on MMLab Toolkits

[sota-cd](https://hyper.ai/cn/sota/tasks/change-detection)

[Benchmark SECOND Dataset](https://github.com/ale93111/awesome-semantic-change-detection?tab=readme-ov-file#benchmark)

[awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection) 

[Change-Detection-Review](https://github.com/MinZHANG-WHU/Change-Detection-Review)

## SECOND DATASET
|               Model               |  mIoU |  SeK  | Score | Status                  |
| :-------------------------------: | :---: | :---: | :---: | :---------------------: |
|                TaCo               | 73.77 | 24.73 | 39.44 | No Code                 |
|               DaCDF               | 72.30 | 21.88 | 37.00 | No Git                  |
|             UniChange             | 72.85 | 23.02 | 37.97 | No Code                 |
|              GSTM-SCD             | 73.61 | 24.36 | 39.13 | Mamba-base, have weight |
|                FoBa               | 74.50 | 24.61 | 39.58 | Mamba-base, have weight |
|              Mamba-FCS            | 74.07 | 25.50 | 40.07 | No Git                  |
|             AWMambaSCD            | 73.66 | 24.95 | 39.56 | No Git                  |
|               BT-SCD              | 73.67 | 24.21 | 39.04 | No Git                  |
|               STGNet              | 72.83 | 22.45 | 37.56 | No Git                  |
| SCanNet + CBAM + L<sub>Dice</sub> | 73.63 | 24.25 | 39.06 | No weight               |
|               SCNet               | 73.85 | 23.99 | 38.95 | wait GDrive Assess      |
|             VFM-ReSCD             | 73.33 | 24.01 | 38.81 | No weight               |
|            Semantic-CD            | 75.10 | 23.85 | 39.23 | No Git                  |
|             MamabaSCD             | 73.68 | 22.92 | 38.15 | Mamba-base, have weight |
|              SCD-SAM              | 77.75 | 32.44 | 46.03 | SAM-base, have weight   |
|              LSAFNet              | 74.01 | 24.32 | 39.23 | No weight               |
|               HGINet              |   --  |   --  |   --  | No weight               |
|            DEFO-MLTSCD            | 73.76 | 23.73 | 38.74 | `PASS`                  |
|               DFINet              | 72.61 | 20.12 | 35.87 | No Git                  |
|              SCanNet              | 73.42 | 23.94 | 38.78 | `PASS`                  |