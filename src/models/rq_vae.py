import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from .encoder import Encoder
from .decoder import Decoder
from .quantize import VectorQuantizer
from .loss import VAELoss
from ..utils.metrics import calculate_metrics

class RQVAE(pl.LightningModule):
    def __init__(self, cfg: DictConfig, num_users: int, num_items: int):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.num_users = num_users
        self.num_items = num_items

        # Model components
        self.encoder = Encoder(
            input_dim=num_items,
            hidden_dim=cfg.model.encoder.hidden_dim,
            latent_dim=cfg.model.encoder.latent_dim,
            num_layers=cfg.model.encoder.num_layers
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=cfg.model.model.num_embeddings,
            embedding_dim=cfg.model.model.embedding_dim,
            commitment_cost=cfg.model.model.commitment_cost,
            decay=cfg.model.model.decay
        )

        self.decoder = Decoder(
            latent_dim=cfg.model.decoder.latent_dim,
            hidden_dim=cfg.model.decoder.hidden_dim,
            output_dim=num_items,
            num_layers=cfg.model.decoder.num_layers
        )

        # Loss function
        self.loss_fn = VAELoss()

        # Metrics
        self.train_metrics = {}
        self.val_metrics = {}

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        # Forward pass through the model
        z = self.encoder(x)
        z_q, vq_loss, perplexity = self.quantizer(z)
        x_recon = self.decoder(z_q)

        return {
            'x_recon': x_recon,
            'z': z,
            'z_q': z_q,
            'vq_loss': vq_loss,
            'perplexity': perplexity
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch['rating']  # Shape: [batch_size, num_items]

        outputs = self(x)
        recon_loss = self.loss_fn(outputs['x_recon'], x)
        total_loss = recon_loss + outputs['vq_loss']

        # Calculate metrics
        metrics = calculate_metrics(outputs['x_recon'], x, k=[10, 20])

        # Logging
        self.log('train/total_loss', total_loss, prog_bar=True, logger=True)
        self.log('train/recon_loss', recon_loss, logger=True)
        self.log('train/vq_loss', outputs['vq_loss'], logger=True)
        self.log('train/perplexity', outputs['perplexity'], logger=True)

        for k, v in metrics.items():
            self.log(f'train/{k}', v, logger=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        x = batch['rating']

        outputs = self(x)
        recon_loss = self.loss_fn(outputs['x_recon'], x)
        total_loss = recon_loss + outputs['vq_loss']

        # Calculate metrics
        metrics = calculate_metrics(outputs['x_recon'], x, k=[10, 20])

        # Logging
        self.log('val/total_loss', total_loss, prog_bar=True, logger=True)
        self.log('val/recon_loss', recon_loss, logger=True)
        self.log('val/vq_loss', outputs['vq_loss'], logger=True)
        self.log('val/perplexity', outputs['perplexity'], logger=True)

        for k, v in metrics.items():
            self.log(f'val/{k}', v, logger=True)

        return {'val_loss': total_loss, **metrics}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        x = batch['rating']

        outputs = self(x)
        recon_loss = self.loss_fn(outputs['x_recon'], x)
        total_loss = recon_loss + outputs['vq_loss']

        # Calculate metrics
        metrics = calculate_metrics(outputs['x_recon'], x, k=[10, 20, 50])

        # Logging
        self.log('test/total_loss', total_loss, logger=True)
        self.log('test/recon_loss', recon_loss, logger=True)
        self.log('test/vq_loss', outputs['vq_loss'], logger=True)
        self.log('test/perplexity', outputs['perplexity'], logger=True)

        for k, v in metrics.items():
            self.log(f'test/{k}', v, logger=True)

        return {'test_loss': total_loss, **metrics}

    def configure_optimizers(self):
        optimizer_config = self.cfg.train.optimizer

        if optimizer_config.name == 'adam':
            optimizer = Adam(
                self.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay
            )
        elif optimizer_config.name == 'adamw':
            optimizer = AdamW(
                self.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.name}")

        scheduler_config = self.cfg.train.scheduler

        if scheduler_config.name == 'step':
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma
            )
        elif scheduler_config.name == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.factor,
                patience=scheduler_config.patience
            )
        else:
            return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss'
            }
        }

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        x = batch['rating']
        outputs = self(x)
        return outputs['x_recon']

class RQVAEDecoder(pl.LightningModule):
    def __init__(self, cfg: DictConfig, num_users: int, num_items: int):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.num_users = num_users
        self.num_items = num_items

        # Decoder only for fine-tuning
        self.decoder = Decoder(
            latent_dim=cfg.model.decoder.latent_dim,
            hidden_dim=cfg.model.decoder.hidden_dim,
            output_dim=num_items,
            num_layers=cfg.model.decoder.num_layers
        )

        # Loss function
        self.loss_fn = nn.MSELoss()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        z = batch['latent']
        x_target = batch['rating']

        x_recon = self(z)
        loss = self.loss_fn(x_recon, x_target)

        # Logging
        self.log('train/decoder_loss', loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        z = batch['latent']
        x_target = batch['rating']

        x_recon = self(z)
        loss = self.loss_fn(x_recon, x_target)

        # Logging
        self.log('val/decoder_loss', loss, prog_bar=True, logger=True)

        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer_config = self.cfg.train.optimizer

        if optimizer_config.name == 'adam':
            optimizer = Adam(
                self.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay
            )
        elif optimizer_config.name == 'adamw':
            optimizer = AdamW(
                self.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.name}")

        return optimizer

@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Test model initialization
    model = RQVAE(cfg, num_users=6040, num_items=3952)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    x = torch.randn(32, 3952)  # Batch of 32, 3952 items
    outputs = model(x)
    print(f"Forward pass complete")
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {outputs['x_recon'].shape}")
    print(f"Latent shape: {outputs['z'].shape}")
    print(f"Quantized latent shape: {outputs['z_q'].shape}")
    print(f"VQ loss: {outputs['vq_loss'].item()}")
    print(f"Perplexity: {outputs['perplexity'].item()}")

if __name__ == "__main__":
    main()
