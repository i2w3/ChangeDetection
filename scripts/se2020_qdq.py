from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, quantize_static
from tqdm import tqdm


def preprocess_model(model_path: Path, output_path: Path) -> None:
    '''预处理模型，进行基本的图优化并保存优化后的模型
    '''
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.optimized_model_filepath = str(output_path)
    _ = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])


def change_parent_dir(file_path: Path, old_parent: str, new_parent: str) -> Path:
    '''更改文件的其中一个父路径，注意仅更改第一个匹配到的父路径部分
    eg: change_parent_dir(./res/data/SE2020_CD/val/im1/00004.png, "im1", "im2") -> ./res/data/SE2020_CD/val/im2/00004.png
    '''
    parts = list(file_path.parts)
    try:
        index = parts.index(old_parent)
        parts[index] = new_parent
        return Path(*parts)
    except ValueError:
        return file_path


class SE2020DataReader(CalibrationDataReader):
    def __init__(self, data_dir: Path | str, limit: int = 100):
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.image_lists = list((self.data_dir / "val" / "im1").glob("*.png"))[:limit]
        self._i = 0 # 计数器
        self._init_pbar()

        self.model_mean = [0.485, 0.456, 0.406]
        self.model_std  = [0.229, 0.224, 0.225]
        self.model_mean_bgr255:list[int] = (self.model_mean[2]*255, 
                                            self.model_mean[1]*255, 
                                            self.model_mean[0]*255)
        self.model_std_array:np.ndarray = np.array(self.model_std).reshape(1, 3, 1, 1).astype(np.float32)

    def get_next(self) -> dict[str, np.ndarray] | None:
        '''获取下一个 ort 格式数据
        '''
        if self._i >= len(self.image_lists):
            self.pbar.close()
            return None
        img_path = self.image_lists[self._i]
        img_1 = cv2.imread(str(img_path))
        img_2 = cv2.imread(str(change_parent_dir(img_path, "im1", "im2")))
        img_1_blob = self.preProcess(img_1)
        img_2_blob = self.preProcess(img_2)
        self._i += 1
        self.pbar.update(1)
        return {"input1": img_1_blob, "input2": img_2_blob}
        
    def preProcess(self, src:np.ndarray) -> np.ndarray:
        '''预处理
        '''
        img = src.copy()
        shape = src.shape[:2]  # [height, width, channel] -> [height, width]
        if (shape[0] != 512 or shape[1] != 512):
            img = cv2.resize(img, (512, 512))
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (512, 512), self.model_mean_bgr255, swapRB=True, crop=False, ddepth=cv2.CV_32F)
        blob /= self.model_std_array # 注意 std 不用切换为 BGR 顺序
        return blob

    def rewind(self) -> None:
        '''重置计数器
        '''
        self._i = 0
        self.pbar.close()
        self._init_pbar()

    def _init_pbar(self) -> None:
        '''初始化进度条
        '''
        self.pbar = tqdm(total=len(self.image_lists), desc="Calibration Progress", unit="img")


if __name__ == "__main__":
    # 原始模型路径
    raw_model = Path("./res/pspnet_hrnet_w40.onnx")
    # 输出优化后模型路径
    cleaned_model = raw_model.parent / f"{raw_model.stem}_cleaned{raw_model.suffix}"
    # 输出量化模型路径
    quantized_model = raw_model.parent / f"{raw_model.stem}_int8_qdq{raw_model.suffix}"

    # 1. 预处理模型
    preprocess_model(raw_model, cleaned_model)

    # 2. 实例化数据读取器
    data_reader = SE2020DataReader(data_dir="./res/data/SE2020_CD")

    # 3. 执行静态量化
    trt_extra_options = {
        'ActivationSymmetric': True,  # 关键：强制激活值为对称量化
        'WeightSymmetric': True,      # 关键：强制权重为对称量化
        'QuantizeBias': False,        # 即使量化 Bias，TensorRT 通常也会忽略或融合，建议 False 以减少算子
        'ForceQuantizeNoInputCheck': True # 某些情况下有助于跳过特定的输入检查
    }

    quantize_static(
        model_input=cleaned_model,
        model_output=quantized_model,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8, 
        weight_type=QuantType.QInt8,
        extra_options=trt_extra_options,
        calibration_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )