"""Behavior cloning imitation learning."""

import torch
import torch.nn as nn


class BCLoss(nn.Module):
    """Behavior cloning loss function."""

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        """Compute BC loss.

        Args:
            predictions: (throttle, brake, steering) predicted
            targets: (throttle, brake, steering) ground truth

        Returns:
            Loss scalar
        """
        throttle_pred, brake_pred, steering_pred = predictions
        throttle_gt, brake_gt, steering_gt = targets

        loss = (
            self.mse_loss(throttle_pred, throttle_gt) +
            self.mse_loss(brake_pred, brake_gt) +
            self.mse_loss(steering_pred, steering_gt)
        ) / 3.0

        return loss
