"""ONNX model export for deployment."""

import os
import torch
from typing import Optional

try:
    import onnx
except ImportError:
    onnx = None


class ONNXExporter:
    """Export PyTorch models to ONNX format."""

    def __init__(self):
        pass

    def export_policy(self, model: torch.nn.Module, output_path: str, input_shape: tuple = (1, 3, 64, 64)):
        """Export policy model to ONNX.

        Args:
            model: PyTorch model
            output_path: Path to save ONNX model
            input_shape: Shape of input tensor
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['rgb_image'],
            output_names=['throttle', 'brake', 'steering'],
            opset_version=11,
        )

        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

    def export_world_model(self, model: torch.nn.Module, output_path: str):
        """Export world model to ONNX.

        Args:
            model: PyTorch world model
            output_path: Path to save ONNX model
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        dummy_image = torch.randn(1, 3, 64, 64)
        dummy_action = torch.randn(1, 3)

        torch.onnx.export(
            model,
            (dummy_image, dummy_action),
            output_path,
            input_names=['current_frame', 'action'],
            output_names=['next_frame'],
            opset_version=11,
        )

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
