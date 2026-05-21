"""World model for future prediction (optional)."""

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """World model that predicts next frame from current frame and action."""

    def __init__(self, input_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc_encode = nn.Linear(128, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim + 3, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next frame.

        Args:
            x: Current frame (batch_size, 3, 64, 64)
            action: Action taken (batch_size, 3)

        Returns:
            Predicted next frame
        """
        # Encode
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        z = self.fc_encode(encoded)

        # Decode with action
        z_action = torch.cat([z, action], dim=1)
        decoded = self.fc_decode(z_action)
        decoded = decoded.view(decoded.size(0), 128, 8, 8)
        output = self.decoder(decoded)

        return output
