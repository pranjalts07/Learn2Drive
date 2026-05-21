"""Loss functions for world model training."""

import torch
import torch.nn as nn


class WorldModelLoss(nn.Module):
    """Loss for world model frame prediction."""

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_frame: torch.Tensor, target_frame: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between predicted and target frames.

        Args:
            predicted_frame: Predicted next frame
            target_frame: Ground truth next frame

        Returns:
            Loss scalar
        """
        return self.mse_loss(predicted_frame, target_frame)
