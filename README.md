# model list
all onnx model can download from [huggingface](https://huggingface.co/i2w3/model_zoo)

## [SE2020](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection)
converted onnx model upload to [huggingface](https://huggingface.co/i2w3/model_zoo)

## [MambaCD](https://github.com/ChenHongruixuan/MambaCD)
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

此外，此模型由于有自定义 cuda 算子，转换为 onnx 后与原始 pt 模型输出存在明显差异

# more model
[open-cd](https://github.com/likyoo/open-cd) base on MMLab Toolkits

[sota-cd](https://hyper.ai/cn/sota/tasks/change-detection)

[awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection) 

[Change-Detection-Review](https://github.com/MinZHANG-WHU/Change-Detection-Review)