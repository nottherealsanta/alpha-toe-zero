# export_onnx_4x4x4.py
import torch
import numpy as np

# Import from your 4x4x4 implementation file
# If you named it differently, adjust the module name below.
from qubic import AlphaZeroModel, AlphaZeroConvNet

ONNX_PATH = "model_4x4x4.onnx"
CKPT_PATH = "models_4x4x4/alphazero_iter_190.pth"  # adjust to your checkpoint

def load_model(ckpt_path: str) -> AlphaZeroConvNet:
    m = AlphaZeroModel.load_from_file(ckpt_path, device='cpu')
    m.net.eval()
    m.net.to('cpu')
    return m.net

def dummy_input() -> torch.Tensor:
    # Shape: [B, C, D, H, W] = [1, 2, 4, 4, 4]
    x = np.zeros((1, 2, 4, 4, 4), dtype=np.float32)
    return torch.from_numpy(x)

def export(net: AlphaZeroConvNet, onnx_path: str):
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