# Makefile for RQ-VAE-Recommender

# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Default target
.DEFAULT_GOAL := help

# Help target
help:
    @echo "RQ-VAE-Recommender"
    @echo ""
    @echo "Available targets:"
    @echo "  install                Install dependencies"
    @echo "  install-dev            Install development dependencies"
    @echo "  train-rqvae            Train RQ-VAE model"
    @echo "  train-decoder          Train decoder only"
    @echo "  eval                   Evaluate model"
    @echo "  test                   Run tests"
    @echo "  lint                   Run linters"
    @echo "  format                 Format code"
    @echo "  clean                  Clean up generated files"
    @echo "  help                   Show this help message"

# Install dependencies
install:
    $(PIP) install -r requirements.txt

# Install development dependencies
install-dev:
    $(PIP) install -r requirements.txt
    $(PIP) install pre-commit black flake8 isort

# Train RQ-VAE model
train-rqvae:
    $(PYTHON) main.py --config-name train_rqvae.yaml

# Train decoder only
train-decoder:
    $(PYTHON) main.py --config-name train_decoder.yaml

# Evaluate model
eval:
    $(PYTHON) main.py --config-name eval.yaml

# Run tests
test:
    $(PYTHON) -m pytest tests/ -v

# Run linters
lint:
    $(PYTHON) -m flake8 src/
    $(PYTHON) -m isort --check-only src/
    $(PYTHON) -m black --check src/

# Format code
format:
    $(PYTHON) -m isort src/
    $(PYTHON) -m black src/

# Clean up generated files
clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
    find . -type d -name "*.egg-info" -delete
    rm -rf logs/
    rm -rf checkpoints/
    rm -rf .pytest_cache/
    rm -rf .hydra/

# Initialize git hooks
init-hooks:
    pre-commit install

# Create directory structure
init-dirs:
    mkdir -p logs/ checkpoints/ data/ tests/

.PHONY: help install install-dev train-rqvae train-decoder eval test lint format clean init-hooks init-dirs