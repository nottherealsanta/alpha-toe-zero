# export_onnx.py
import torch
import numpy as np

# import your classes
from main import AlphaZeroModel, AlphaZeroNet  # adjust import

ONNX_PATH = "models/model.onnx"
CKPT_PATH = "models/alphazero_iter_40.pth"  # or any .pth you saved

def load_model(ckpt_path):
    m = AlphaZeroModel.load_from_file(ckpt_path, device='cpu')
    m.net.eval()
    return m.net

def dummy_input():
    # shape [B,3,3,2], float32
    x = np.zeros((1,3,3,2), dtype=np.float32)
    return torch.from_numpy(x)

def export(net, onnx_path):
    x = dummy_input()
    torch.onnx.export(
        net,
        x,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["value", "policy_logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "value": {0: "batch"},
            "policy_logits": {0: "batch"},
        },
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    net = load_model(CKPT_PATH)
    export(net, ONNX_PATH)