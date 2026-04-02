#!/usr/bin/env python3
"""
Setup script for RQ-VAE-Recommender environment
"""

import os
import sys
import subprocess
import venv

def create_virtual_environment():
    """Create virtual environment if it doesn't exist"""
    if not os.path.exists('.venv'):
        print("Creating virtual environment...")
        venv.create('.venv', with_pip=True)
        print("✓ Virtual environment created")
    else:
        print("✓ Virtual environment already exists")

def install_dependencies():
    """Install required dependencies"""
    pip_path = os.path.join('.venv', 'bin', 'pip') if sys.platform != 'win32' else os.path.join('.venv', 'Scripts', 'pip.exe')

    print("\nInstalling dependencies...")
    subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
    print("✓ Dependencies installed")

def install_development_dependencies():
    """Install development dependencies"""
    pip_path = os.path.join('.venv', 'bin', 'pip') if sys.platform != 'win32' else os.path.join('.venv', 'Scripts', 'pip.exe')

    print("\nInstalling development dependencies...")
    subprocess.run([pip_path, 'install', 'pre-commit', 'black', 'flake8', 'isort', 'pytest'], check=True)
    print("✓ Development dependencies installed")

def setup_pre_commit_hooks():
    """Setup pre-commit hooks"""
    pre_commit_path = os.path.join('.venv', 'bin', 'pre-commit') if sys.platform != 'win32' else os.path.join('.venv', 'Scripts', 'pre-commit.exe')

    print("\nSetting up pre-commit hooks...")
    subprocess.run([pre_commit_path, 'install'], check=True)
    print("✓ Pre-commit hooks installed")

def create_directories():
    """Create required directories"""
    directories = ['logs', 'checkpoints', 'data', 'tests', 'notebooks']

    print("\nCreating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}/")

def main():
    """Main setup function"""
    print("=" * 50)
    print("RQ-VAE-Recommender Setup")
    print("=" * 50)

    create_virtual_environment()
    install_dependencies()

    while True:
        response = input("\nInstall development dependencies? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            install_development_dependencies()
            setup_pre_commit_hooks()
            break
        elif response in ['n', 'no']:
            break
        else:
            print("Please enter 'y' or 'n'")

    create_directories()

    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print("\nTo activate the virtual environment:")
    print("  source .venv/bin/activate")
    print("\nTo train the model:")
    print("  make train-rqvae")
    print("  ")
    print("Or using python:")
    print("  python main.py --config-name train_rqvae.yaml")

if __name__ == "__main__":
    main()
