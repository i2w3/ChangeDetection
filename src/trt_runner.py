import json
from dataclasses import dataclass

import numpy as np

# @dataclass
# class Binding:
#     name: str
#     dtype: np.ndarray
#     shape: tuple[int, ...]
    # data:
    # prt:

from collections import OrderedDict, namedtuple

import torch
import tensorrt as trt

device = torch.device("cuda:0")
Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
logger = trt.Logger(trt.Logger.INFO)
# Read file


w = "./res/trt_cache/TensorrtExecutionProvider_TRTKernel_graph_main_graph_5986799580881147499_0_0_fp16_sm89.engine"
with open(w, "rb") as f, trt.Runtime(logger) as runtime:
    f.seek(0) # go to beginning
    model = runtime.deserialize_cuda_engine(f.read())  # read engine

# Model context
try:
    context = model.create_execution_context()
except Exception as e:  # model is None
    raise e

bindings = OrderedDict()
output_names = []
fp16 = False  # default updated below
dynamic = False
is_trt10 = not hasattr(model, "num_bindings")
num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
for i in num:
    # Get tensor info using TRT10+ or legacy API
    if is_trt10:
        name = model.get_tensor_name(i)
        dtype = trt.nptype(model.get_tensor_dtype(name))
        is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        shape = tuple(model.get_tensor_shape(name))
        profile_shape = tuple(model.get_tensor_profile_shape(name, 0)[2]) if is_input else None
    else:
        name = model.get_binding_name(i)
        dtype = trt.nptype(model.get_binding_dtype(i))
        is_input = model.binding_is_input(i)
        shape = tuple(model.get_binding_shape(i))
        profile_shape = tuple(model.get_profile_shape(0, i)[1]) if is_input else None

    # Process input/output tensors
    if is_input:
        if -1 in shape:
            dynamic = True
            print(f"Dynamic shape model detected, setting input '{name}' to profile shape {profile_shape}")
            if is_trt10:
                context.set_input_shape(name, profile_shape)
            else:
                context.set_binding_shape(i, profile_shape)
        if dtype == np.float16:
            print(f"Input '{name}' is fp16, enabling fp16 mode.")
            fp16 = True
    else:
        output_names.append(name)
    shape = tuple(context.get_tensor_shape(name)) if is_trt10 else tuple(context.get_binding_shape(i))
    im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
    bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
print(binding_addrs)