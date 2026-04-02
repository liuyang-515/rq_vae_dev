# RQ-VAE Recommender

This is a PyTorch implementation of a generative retrieval model using semantic IDs based on RQ-VAE from "Recommender Systems with Generative Retrieval".

The project has been refactored to follow the **lightning-hydra-template** structure, making it more modular, maintainable, and scalable.

## Model Overview
The model has two stages:
1. Items in the corpus are mapped to a tuple of semantic IDs by training an RQ-VAE
2. Sequences of semantic IDs are tokenized by using a frozen RQ-VAE and a transformer-based model is trained on sequences of semantic IDs to generate the next ids in the sequence.

![image](https://github.com/EdoardoBotta/RQ-VAE/assets/64335373/199b38ac-a282-4ba1-bd89-3291617e6aa5)

### Currently supports
* **Datasets:** Amazon Reviews (Beauty, Sports, Toys), MovieLens 1M, MovieLens 32M
* RQ-VAE PyTorch model implementation + KMeans initialization + Training script
* Decoder-only retrieval model + Training code with semantic id user sequences
* PyTorch Lightning integration for easy training and scaling
* Hydra configuration system for flexible parameter management

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd RQ-VAE-Recommender
```

2. Create virtual environment and install dependencies:
```bash
# Option 1: Using setup script
python scripts/setup_environment.py

# Option 2: Manual setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Install development dependencies (optional but recommended):
```bash
pip install -r requirements-dev.txt  # Create this file if needed
pre-commit install
```

## 🚀 Usage

### Project Structure
```
RQ-VAE-Recommender/
├── src/
│   ├── data/              # Data loading and processing
│   │   ├── datamodule.py  # LightningDataModule implementation
│   │   └── ...           # Dataset-specific code
│   ├── models/           # Model implementations
│   │   ├── rq_vae.py     # Main RQ-VAE LightningModule
│   │   ├── decoder.py    # Decoder-only model
│   │   └── ...           # Model components
│   ├── utils/            # Utility functions
│   └── __init__.py
├── configs/              # Hydra configuration files
│   ├── train/           # Training configurations
│   ├── model/           # Model configurations
│   ├── data/            # Data configurations
│   ├── config.yaml      # Main configuration
│   ├── train_rqvae.yaml # RQ-VAE training task
│   ├── train_decoder.yaml # Decoder training task
│   └── eval.yaml        # Evaluation configuration
├── scripts/             # Helper scripts
├── tests/               # Test files
├── logs/                # Training logs (generated automatically)
├── checkpoints/         # Model checkpoints (generated automatically)
├── main.py              # Main entry point
├── Makefile             # Makefile for common tasks
├── pyproject.toml       # Project metadata
└── README.md
```

### Training

#### Train RQ-VAE Model
```bash
# Using Makefile
make train-rqvae

# Using Python directly
python main.py --config-name train_rqvae.yaml

# With custom configuration overrides
python main.py --config-name train_rqvae.yaml train.batch_size=512 model.latent_dim=128
```

#### Train Decoder-Only Model
```bash
# Using Makefile
make train-decoder

# Using Python directly
python main.py --config-name train_decoder.yaml
```

### Evaluation
```bash
# Evaluate trained model
python main.py --config-name eval.yaml eval.checkpoint.path=./checkpoints/best_model.ckpt
```

### Configuration

All configuration is handled via Hydra. You can customize training by:

1. Modifying the YAML files in `configs/` directory
2. Using command-line overrides
3. Creating new configuration files

Example configuration structure:
```yaml
task:
  name: train_rqvae

train:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001

model:
  model:
    name: rq_vae
    latent_dim: 64
    hidden_dim: 256
    num_embeddings: 512

model:
  dataset:
    name: ml-1m
    path: ./data/ml-1m
```

### Monitoring

Training logs are stored in `logs/` directory with timestamped subdirectories. You can monitor training using:

#### TensorBoard
```bash
tensorboard --logdir logs/
```

#### Weights & Biases
If you have wandb installed, training will automatically log to your wandb account.

## 📊 Results

### Checkpoints
Model checkpoints are saved in `checkpoints/` directory:
- Best models based on validation loss
- Last checkpoint from training

### Metrics
The model tracks various metrics during training:
- Reconstruction loss
- VQ loss
- Perplexity
- NDCG, Precision, Recall, MAP at various k values

## 🔧 Development

### Testing
```bash
make test
# or
python -m pytest tests/ -v
```

### Linting and Formatting
```bash
# Check for issues
make lint

# Auto-format code
make format
```

### Cleaning Up
```bash
make clean
```

## 📚 References

* [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) by Shashank Rajput et al.
* [Restructuring Vector Quantization with the Rotation Trick](https://arxiv.org/abs/2410.06424) by Christopher Fifty et al.
* [PyTorch Lightning](https://www.pytorchlightning.ai/) - PyTorch wrapper for high-performance training
* [Hydra](https://hydra.cc/) - Configuration framework

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
