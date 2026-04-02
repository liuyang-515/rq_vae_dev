import pytest
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.models.rq_vae import RQVAE, RQVAEDecoder
from src.data.datamodule import RecommenderDataModule

@pytest.fixture
def config():
    """Create a minimal configuration for testing"""
    cfg = DictConfig({
        'data': {
            'dataset': {
                'name': 'ml-1m',
                'path': './data/ml-1m'
            },
            'dataloader': {
                'batch_size': 32,
                'num_workers': 0,
                'shuffle': True,
                'pin_memory': False
            },
            'preprocessing': {
                'normalize': True,
                'min_rating': 1,
                'max_rating': 5
            }
        },
        'model': {
            'model': {
                'name': 'rq_vae',
                'latent_dim': 64,
                'hidden_dim': 64,
                'num_layers': 2,
                'num_embeddings': 64,
                'embedding_dim': 64,
                'commitment_cost': 0.25,
                'decay': 0.99
            },
            'encoder': {
                'name': 'rq_encoder',
                'input_dim': 3952,
                'hidden_dim': 64,
                'latent_dim': 64,
                'num_layers': 2
            },
            'decoder': {
                'latent_dim': 64,
                'hidden_dim': 64,
                'output_dim': 3952,
                'num_layers': 2
            }
        },
        'train': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 10,
            'optimizer': {
                'name': 'adam',
                'lr': 0.001,
                'weight_decay': 0.0001
            },
            'scheduler': {
                'name': 'step',
                'step_size': 30,
                'gamma': 0.1
            },
            'logging': {
                'checkpoint_dir': './checkpoints',
                'log_dir': './logs',
                'save_top_k': 3,
                'monitor': 'val_loss'
            }
        },
        'seed': 42
    })
    return cfg

def test_rq_vae_init(config):
    """Test RQ-VAE model initialization"""
    model = RQVAE(config, num_users=6040, num_items=3952)

    # Check model components exist
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'quantizer')
    assert hasattr(model, 'decoder')

    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0

    print(f"RQ-VAE model initialized with {total_params:,} parameters")

def test_rq_vae_forward(config):
    """Test RQ-VAE forward pass"""
    model = RQVAE(config, num_users=6040, num_items=3952)

    # Create dummy input
    batch_size = 32
    num_items = 3952
    x = torch.randn(batch_size, num_items)

    # Forward pass
    outputs = model(x)

    # Check outputs
    assert 'x_recon' in outputs
    assert 'z' in outputs
    assert 'z_q' in outputs
    assert 'vq_loss' in outputs
    assert 'perplexity' in outputs

    # Check shapes
    assert outputs['x_recon'].shape == (batch_size, num_items)
    assert outputs['z'].shape == (batch_size, config.model.model.latent_dim)
    assert outputs['z_q'].shape == (batch_size, config.model.model.latent_dim)

    # Check values are valid
    assert not torch.isnan(outputs['x_recon']).any()
    assert not torch.isnan(outputs['vq_loss'])
    assert outputs['perplexity'] > 0

    print("RQ-VAE forward pass successful")

def test_rq_vae_decoder_init(config):
    """Test RQ-VAE decoder only initialization"""
    model = RQVAEDecoder(config, num_users=6040, num_items=3952)

    # Check model components exist
    assert hasattr(model, 'decoder')

    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0

    print(f"RQ-VAE decoder initialized with {total_params:,} parameters")

def test_rq_vae_decoder_forward(config):
    """Test RQ-VAE decoder forward pass"""
    model = RQVAEDecoder(config, num_users=6040, num_items=3952)

    # Create dummy input
    batch_size = 32
    latent_dim = config.model.model.latent_dim
    z = torch.randn(batch_size, latent_dim)

    # Forward pass
    x_recon = model(z)

    # Check output shape
    assert x_recon.shape == (batch_size, 3952)

    # Check values are valid
    assert not torch.isnan(x_recon).any()

    print("RQ-VAE decoder forward pass successful")

def test_model_training_step(config):
    """Test model training step"""
    model = RQVAE(config, num_users=6040, num_items=3952)

    # Create dummy batch
    batch_size = 32
    num_items = 3952

    batch = {
        'user_id': torch.randint(0, 6040, (batch_size,)),
        'item_id': torch.randint(0, 3952, (batch_size,)),
        'rating': torch.randn(batch_size, num_items)
    }

    # Training step
    loss = model.training_step(batch, batch_idx=0)

    # Check loss is valid
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss > 0

    print(f"Training step successful, loss: {loss.item():.4f}")

def test_model_validation_step(config):
    """Test model validation step"""
    model = RQVAE(config, num_users=6040, num_items=3952)

    # Create dummy batch
    batch_size = 32
    num_items = 3952

    batch = {
        'user_id': torch.randint(0, 6040, (batch_size,)),
        'item_id': torch.randint(0, 3952, (batch_size,)),
        'rating': torch.randn(batch_size, num_items)
    }

    # Validation step
    outputs = model.validation_step(batch, batch_idx=0)

    # Check outputs
    assert 'val_loss' in outputs
    assert not torch.isnan(outputs['val_loss'])
    assert outputs['val_loss'] > 0

    print(f"Validation step successful, loss: {outputs['val_loss'].item():.4f}")

@pytest.mark.parametrize("batch_size", [1, 32, 64])
def test_model_batch_sizes(config, batch_size):
    """Test model with different batch sizes"""
    model = RQVAE(config, num_users=6040, num_items=3952)

    # Create dummy input
    num_items = 3952
    x = torch.randn(batch_size, num_items)

    # Forward pass
    outputs = model(x)

    # Check output shapes
    assert outputs['x_recon'].shape == (batch_size, num_items)
    assert outputs['z'].shape == (batch_size, config.model.model.latent_dim)

    print(f"Batch size {batch_size} test successful")

def test_config_override(config):
    """Test model with configuration overrides"""
    # Modify configuration
    config.model.model.latent_dim = 128
    config.model.encoder.hidden_dim = 128
    config.model.decoder.hidden_dim = 128

    model = RQVAE(config, num_users=6040, num_items=3952)

    # Create dummy input
    batch_size = 32
    num_items = 3952
    x = torch.randn(batch_size, num_items)

    # Forward pass
    outputs = model(x)

    # Check latent dimension changed
    assert outputs['z'].shape == (batch_size, 128)

    print("Configuration override test successful")

if __name__ == "__main__":
    # Run tests manually
    config = DictConfig({
        'data': {
            'dataset': {
                'name': 'ml-1m',
                'path': './data/ml-1m'
            },
            'dataloader': {
                'batch_size': 32,
                'num_workers': 0,
                'shuffle': True,
                'pin_memory': False
            },
            'preprocessing': {
                'normalize': True,
                'min_rating': 1,
                'max_rating': 5
            }
        },
        'model': {
            'model': {
                'name': 'rq_vae',
                'latent_dim': 64,
                'hidden_dim': 64,
                'num_layers': 2,
                'num_embeddings': 64,
                'embedding_dim': 64,
                'commitment_cost': 0.25,
                'decay': 0.99
            },
            'encoder': {
                'name': 'rq_encoder',
                'input_dim': 3952,
                'hidden_dim': 64,
                'latent_dim': 64,
                'num_layers': 2
            },
            'decoder': {
                'latent_dim': 64,
                'hidden_dim': 64,
                'output_dim': 3952,
                'num_layers': 2
            }
        },
        'train': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 10,
            'optimizer': {
                'name': 'adam',
                'lr': 0.001,
                'weight_decay': 0.0001
            },
            'scheduler': {
                'name': 'step',
                'step_size': 30,
                'gamma': 0.1
            },
            'logging': {
                'checkpoint_dir': './checkpoints',
                'log_dir': './logs',
                'save_top_k': 3,
                'monitor': 'val_loss'
            }
        },
        'seed': 42
    })

    print("Running model tests...")
    test_rq_vae_init(config)
    test_rq_vae_forward(config)
    test_rq_vae_decoder_init(config)
    test_rq_vae_decoder_forward(config)
    test_model_training_step(config)
    test_model_validation_step(config)
    test_model_batch_sizes(config, 32)
    test_config_override(config)
    print("\nAll tests passed!")
