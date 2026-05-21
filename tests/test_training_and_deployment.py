"""Test training and deployment utilities."""

import pytest
import tempfile
import torch
from autonomy_stack.training.world_model_losses import WorldModelLoss
from autonomy_stack.deployment.onnx_export import ONNXExporter
from autonomy_stack.models.policy import DrivingPolicy
from autonomy_stack.data_engine.cards import ModelCard, DatasetCard


def test_world_model_loss():
    """Test world model loss."""
    loss_fn = WorldModelLoss()
    predicted = torch.randn(4, 3, 64, 64)
    target = torch.randn(4, 3, 64, 64)
    loss = loss_fn(predicted, target)
    assert loss.item() > 0


def test_onnx_export():
    """Test ONNX export (if ONNX available)."""
    try:
        import onnx
        exporter = ONNXExporter()
        policy = DrivingPolicy()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/policy.onnx"
            exporter.export_policy(policy, output_path)
            assert True  # Export succeeded
    except ImportError:
        pytest.skip("ONNX not installed")


def test_model_card():
    """Test model card."""
    card = ModelCard("test_model", "0.1.0")
    metadata = card.to_dict()
    assert metadata['model_name'] == "test_model"
    assert metadata['version'] == "0.1.0"


def test_dataset_card():
    """Test dataset card."""
    card = DatasetCard("test_dataset", "0.1.0")
    metadata = card.to_dict()
    assert metadata['dataset_name'] == "test_dataset"
    assert metadata['version'] == "0.1.0"


def test_model_card_save():
    """Test model card save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        card = ModelCard("test", "0.1.0")
        path = f"{tmpdir}/card.json"
        card.save(path)
        assert True  # Save succeeded


def test_dataset_card_save():
    """Test dataset card save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        card = DatasetCard("test", "0.1.0")
        path = f"{tmpdir}/card.json"
        card.save(path)
        assert True  # Save succeeded
