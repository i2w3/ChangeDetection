# TODO
- [x] 实现 [SE2020](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection) 的 ONNX 格式转换和 TensorRT 引擎导出，并完成推理流程
- [x] 实现 SE2020 在 GVLM_CD 数据集上推理(SE2020 的训练数据与 GVLM_CD 数据集不匹配，仅用来测试功能./test_gvlm_dataset.py)
- [ ] 模拟 UAV 第一次飞行与第二次飞行不一定匹配的问题(目前使用有大图的遥感数据集 [GVLM_CD](https://www.kaggle.com/datasets/gbhavi/gvlm-change-detection)，方便在上面随机裁剪然后进行匹配，但是遥感图像拍摄年代差别较久，图像上的特征点难以匹配)

## model download
[huggingface](https://huggingface.co/i2w3/model_zoo)