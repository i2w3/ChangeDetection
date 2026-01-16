import torch
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(encoder_name = "resnet50",
                          encoder_weights = "imagenet",
                          decoder_atrous_rates = (12,18,24),
                          encoder_output_stride =16,
                          classes = 5).eval()
model.load_state_dict(torch.load("checkpoint.pt"))

dummy_input = torch.randn(1, 3, 512, 512)
output = model(dummy_input)
print(output.shape)


onnx_program = torch.onnx.export(
    model, 
    (dummy_input,), 
    "./deeplabv3+_LandcoverAI.onnx",
    export_params=True,
    input_names=['input',],
    output_names=['output_bin',],
    opset_version=17,
    external_data=False,
    verbose=False
)

jit_model = torch.jit.trace(model, (dummy_input,))

onnx_program = torch.onnx.export(
    jit_model, 
    (dummy_input,),
    "./deeplabv3+_LandcoverAI_jit.onnx",
    export_params=True,
    input_names=['input',],
    output_names=['output_bin',],
    opset_version=17,
    external_data=False,
    verbose=False
)