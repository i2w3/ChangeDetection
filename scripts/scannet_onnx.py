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