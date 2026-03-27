import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphContrastiveAEClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

        # Binary classifier head
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, dim=1)
        x_recon = self.decoder(z)
        logits = self.classifier(z).squeeze(1)  # (N,)
        return z, x_recon, logits
