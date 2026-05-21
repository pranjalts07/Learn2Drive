"""CNN-based driving policy models: compact baseline and medium architecture."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from autonomy_stack.schemas import ControlAction


class CompactCNNPolicy(nn.Module):
    """Lightweight CNN policy for fast inference and baseline comparison.

    Architecture: RGB image input → shallow CNN encoder → 3 output heads
    Parameters: ~604K total
    Use case: Smoke tests, latency benchmarking, fast local validation
    """

    def __init__(self, input_channels: int = 3, img_height: int = 64, img_width: int = 64):
        super().__init__()

        self.input_channels = input_channels
        self.img_height = img_height
        self.img_width = img_width

        # CNN encoder
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.relu3 = nn.ReLU(inplace=True)

        # Calculate flattened size after convolutions
        with torch.no_grad():
            x = torch.zeros(1, input_channels, img_height, img_width)
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            flattened_size = x.view(1, -1).shape[1]

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU(inplace=True)

        # Output heads
        self.throttle_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.brake_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.steering_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 64, 64)

        Returns:
            Tuple of (throttle, brake, steering) tensors
        """
        # CNN encoding
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)

        # Output heads
        throttle = self.throttle_head(x)
        brake = self.brake_head(x)
        steering = self.steering_head(x)

        return throttle, brake, steering

    def predict(self, rgb_image: np.ndarray) -> ControlAction:
        """Predict control action from RGB image.

        Args:
            rgb_image: RGB image array (H, W, 3) with values in [0, 1] or [0, 255]

        Returns:
            ControlAction with throttle, brake, steering
        """
        self.eval()
        with torch.no_grad():
            # Preprocess image
            if not isinstance(rgb_image, np.ndarray):
                rgb_image = np.array(rgb_image)

            if rgb_image.dtype != np.float32:
                rgb_image = rgb_image.astype(np.float32)
                if rgb_image.max() > 1.0:
                    rgb_image = rgb_image / 255.0

            # Transpose to (C, H, W) and add batch dimension
            x = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)

            # Forward pass
            throttle, brake, steering = self.forward(x)

            action = ControlAction(
                throttle=throttle.item(),
                brake=brake.item(),
                steering=steering.item()
            )
        return action

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MediumCNNPolicy(nn.Module):
    """Medium-sized CNN policy for model-size and latency tradeoff analysis.

    Architecture: RGB image input → deeper CNN encoder → larger FC layers → 3 output heads
    Parameters: ~2.2M total
    Use case: Portfolio model, deployment tradeoff demonstration
    """

    def __init__(self, input_channels: int = 3, img_height: int = 64, img_width: int = 64):
        super().__init__()

        self.input_channels = input_channels
        self.img_height = img_height
        self.img_width = img_width

        # CNN encoder with more capacity
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        # Calculate flattened size after convolutions
        with torch.no_grad():
            x = torch.zeros(1, input_channels, img_height, img_width)
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.relu4(x)
            flattened_size = x.view(1, -1).shape[1]

        # Fully connected layers with more capacity
        self.fc1 = nn.Linear(flattened_size, 512)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 256)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(256, 128)
        self.relu_fc3 = nn.ReLU(inplace=True)

        # Output heads
        self.throttle_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.brake_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.steering_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)

        throttle = self.throttle_head(x)
        brake = self.brake_head(x)
        steering = self.steering_head(x)

        return throttle, brake, steering

    def predict(self, rgb_image: np.ndarray) -> ControlAction:
        """Predict control action from RGB image."""
        self.eval()
        with torch.no_grad():
            if not isinstance(rgb_image, np.ndarray):
                rgb_image = np.array(rgb_image)

            if rgb_image.dtype != np.float32:
                rgb_image = rgb_image.astype(np.float32)
                if rgb_image.max() > 1.0:
                    rgb_image = rgb_image / 255.0

            x = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)
            throttle, brake, steering = self.forward(x)

            action = ControlAction(
                throttle=throttle.item(),
                brake=brake.item(),
                steering=steering.item()
            )
        return action

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


DrivingPolicy = CompactCNNPolicy
