from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # 这行代码会自动管理 CUDA 上下文
import pycuda.driver as cuda


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
    

class SE2020EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir: Path | str, onnx_model_path: Path | str, ecache_path: Path | str, batch_size: int = 8, limit: int = 100):
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if isinstance(onnx_model_path, str):
            onnx_model_path = Path(onnx_model_path)
        if isinstance(ecache_path, str):
            ecache_path = Path(ecache_path)
        self.data_dir = data_dir
        self.cache_file = ecache_path / f"{onnx_model_path.stem}.cache"
        self.batch_size = batch_size
        self.image_lists = list((self.data_dir / "val" / "im1").glob("*.png"))[:limit]
        if len(self.image_lists) == 0:
            raise ValueError(f"check data_dir: {data_dir}, no images found.")

        self.count = 0
        self.current_index = 0
        
        self.input_shape = (self.batch_size, 3, 512, 512)
        self.model_mean = [0.485, 0.456, 0.406]
        self.model_std  = [0.229, 0.224, 0.225]
        self.model_mean_bgr255:list[int] = (self.model_mean[2]*255, 
                                            self.model_mean[1]*255, 
                                            self.model_mean[0]*255)
        self.model_std_array:np.ndarray = np.array(self.model_std).reshape(1, 3, 1, 1).astype(np.float32)

        # 分配单个 batch 需要的 GPU 显存：(B,3,H,W) * 4Bytes (float32)
        self.one_batch_size = trt.volume(self.input_shape) * 4
        # 申请两个输入的显存 (Input1 和 Input2)
        self.d_input1 = cuda.mem_alloc(self.one_batch_size)
        self.d_input2 = cuda.mem_alloc(self.one_batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """TensorRT 会调用此函数来获取一个 Batch 的数据, names 参数用来 debug 返回值顺序
        """
        if self.current_index + self.batch_size > len(self.image_lists):
            # drop last incomplete batch
            return None

        # CPU Buffer (NCHW)
        batch_input1 = np.zeros(self.input_shape, dtype=np.float32)
        batch_input2 = np.zeros(self.input_shape, dtype=np.float32)

        # build one Batch
        for i in range(self.batch_size):
            idx = self.current_index + i
            img_path = self.image_lists[idx]
            
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            img2_path = change_parent_dir(img_path, "im1", "im2")
            if not img2_path.exists():
                raise FileNotFoundError(f"Paired image file not found: {img2_path}")

            img_1_src = cv2.imread(str(img_path))
            img_2_src = cv2.imread(str(img2_path))
            
            batch_input1[i] = self.preProcess(img_1_src)
            batch_input2[i] = self.preProcess(img_2_src)

        self.current_index += self.batch_size

        # CPU -> GPU, 注意使用 np.ascontiguousarray 确保内存连续，防止拷贝出错
        cuda.memcpy_htod(self.d_input1, np.ascontiguousarray(batch_input1))
        cuda.memcpy_htod(self.d_input2, np.ascontiguousarray(batch_input2))

        # 返回显存指针列表
        return [int(self.d_input1), int(self.d_input2)]

    def preProcess(self, src:np.ndarray) -> np.ndarray:
        '''预处理
        '''
        img = src.copy()
        shape = src.shape[:2]  # [height, width, channel] -> [height, width]
        if (shape[0] != 512 or shape[1] != 512):
            img = cv2.resize(img, (512, 512))
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, (512, 512), self.model_mean_bgr255, swapRB=True, crop=False, ddepth=cv2.CV_32F)
        blob /= self.model_std_array # 注意 std 不用切换为 BGR 顺序
        return blob[0] # 要组装 batch，先返回 (3, H, W)

    def read_calibration_cache(self):
        # 如果已有缓存文件，直接读取，这样可以跳过耗时的校准过程
        if Path(self.cache_file).exists():
            print(f"Reading calibration cache from {self.cache_file}")
            return open(self.cache_file, "rb").read()
        return None

    def write_calibration_cache(self, cache):
        # 将生成的校准表写入文件
        print(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_calib_cache():
    # 1. 准备路径
    onnx_model_path = "./res/pspnet_hrnet_w18.onnx"
    onnx_trt_engine_cache_path = "./res/trt_cache"
    data_dir = "./res/data/SE2020_CD"

    # 2. 初始化 TensorRT Builder
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)

    # 3. 解析 ONNX
    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse ONNX')
            return

    # 4. 设置 INT8 和 校准器
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = SE2020EntropyCalibrator(data_dir, onnx_model_path, onnx_trt_engine_cache_path, batch_size=4)

    # 5. 构建 Engine (触发 get_batch 并生成 cache 文件，但是生成的 engine 不保存，使用 ort 来再生成一次)
    # TODO: 实现 trt_runner.py 来直接加载 TensorRT engine 进行推理
    plan = builder.build_serialized_network(network, config)
    
    if plan:
        with open("./demo.engine", "wb") as f:
            f.write(plan)
        print("Build success!")
    else:
        print("Build failed!")
    '''
    TensorrtExecutionProvider::trt_int8_enable = True
                             ::trt_fp16_enable = False
                             ::trt_int8_calibration_table_name = f"{onnx_model_path.stem}.cache"
                             ::trt_int8_use_native_calibration_table = True
    '''

if __name__ == "__main__":
    build_calib_cache()