"""Offline reinforcement learning (TD3+BC style)."""

import torch
import torch.nn as nn


class TD3BCLoss(nn.Module):
    """TD3+BC loss combining temporal difference and behavior cloning."""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, td_loss, bc_loss):
        """Compute combined loss.

        Args:
            td_loss: Temporal difference loss from Q-function
            bc_loss: Behavior cloning loss

        Returns:
            Combined loss
        """
        loss = td_loss + self.alpha * bc_loss
        return loss
