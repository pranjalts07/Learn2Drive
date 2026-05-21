"""Latency profiling and benchmarking utilities."""

import time
import numpy as np
from typing import List, Dict
import torch


class LatencyProfiler:
    """Profile and benchmark inference latency."""

    def __init__(self):
        self.latencies = []

    def profile_inference(self, model: torch.nn.Module, input_shape: tuple, num_runs: int = 100) -> Dict[str, float]:
        """Profile model inference latency.

        Args:
            model: PyTorch model
            input_shape: Shape of input (batch_size, channels, height, width)
            num_runs: Number of runs for averaging

        Returns:
            Dictionary with p50, p95, mean, and fps
        """
        model.eval()
        latencies = []

        with torch.no_grad():
            dummy_input = torch.randn(*input_shape)

            # Warmup
            for _ in range(10):
                _ = model(dummy_input)

            # Measure
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)

        latencies = np.array(latencies)
        self.latencies.extend(latencies)

        results = {
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'mean_ms': float(np.mean(latencies)),
            'fps': float(1000.0 / np.mean(latencies)),
        }

        return results

    def get_model_size_mb(self, model: torch.nn.Module) -> float:
        """Get model size in MB.

        Args:
            model: PyTorch model

        Returns:
            Model size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.data.numel() * 4  # 4 bytes per float32

        return param_size / (1024 * 1024)

    def benchmark_report(self, model: torch.nn.Module, input_shape: tuple = (1, 3, 64, 64)) -> str:
        """Generate benchmark report.

        Args:
            model: PyTorch model
            input_shape: Input shape

        Returns:
            Formatted benchmark report
        """
        latency_results = self.profile_inference(model, input_shape)
        model_size = self.get_model_size_mb(model)
        param_count = sum(p.numel() for p in model.parameters())

        report = f"""
Latency Benchmark Report
========================
Model Size:        {model_size:.2f} MB
Parameters:        {param_count:,}
Inference Latency:
  - p50:           {latency_results['p50_ms']:.2f} ms
  - p95:           {latency_results['p95_ms']:.2f} ms
  - mean:          {latency_results['mean_ms']:.2f} ms
  - fps:           {latency_results['fps']:.1f} FPS
"""
        return report
