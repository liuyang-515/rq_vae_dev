import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Create layers
        layers = []
        input_dim = latent_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

class VariationalDecoder(Decoder):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__(latent_dim, hidden_dim, output_dim, num_layers)

        # Separate mu and logvar heads for variational decoder
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_logvar = nn.Linear(hidden_dim, output_dim)

        # Replace last layer in decoder
        self.decoder = nn.Sequential(*list(self.decoder.children())[:-2])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std