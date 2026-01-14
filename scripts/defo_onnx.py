import torch

from models.TBFFNet import TBFFNet

num_classes = 7


if __name__ == '__main__':
    model = TBFFNet(3, num_classes)
    model.load_state_dict(torch.load('./TBFFNet_49e_mIoU73.76_Sek23.73_Fscd62.73_OA87.80.pth', weights_only=False))
    model.eval()

    # dummy_input1 = torch.randn(1, 3, 512, 512)
    # dummy_input2 = torch.randn(1, 3, 512, 512)

    # onnx_program = torch.onnx.export(
    #     model, 
    #     (dummy_input1, dummy_input2), 
    #     "./DEFO_SECOND.onnx",
    #     export_params=True,
    #     input_names=['input1', 'input2'],
    #     output_names=['output_bin', 'output1', 'output2'],
    #     opset_version=17,
    #     external_data=False,
    #     verbose=False
    # )

    # jit_model = torch.jit.trace(model, (dummy_input1, dummy_input2))

    # onnx_program = torch.onnx.export(
    #     jit_model, 
    #     (dummy_input1, dummy_input2), 
    #     "./DEFO_SECOND_jit.onnx",
    #     export_params=True,
    #     input_names=['input1', 'input2'],
    #     output_names=['output_bin', 'output1', 'output2'],
    #     opset_version=17,
    #     external_data=False,
    #     verbose=False
    # )

    model = model.cuda()