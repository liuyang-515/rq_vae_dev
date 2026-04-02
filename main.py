import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
import torch
import os

from src.data.datamodule import RecommenderDataModule
from src.models.rq_vae import RQVAE, RQVAEDecoder

def set_seed(seed: int):
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def configure_callbacks(cfg: DictConfig):
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',
        dirpath='./checkpoints',
        filename='rq-vae-{epoch:02d}-{val/total_loss:.4f}',
        save_top_k=cfg.train.logging.save_top_k,
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.train.get('early_stopping', None):
        early_stop_callback = EarlyStopping(
            monitor='val/total_loss',
            min_delta=0.00,
            patience=cfg.train.early_stopping.patience,
            verbose=False,
            mode='min'
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    return callbacks

def configure_loggers(cfg: DictConfig):
    loggers = []

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir='./logs',
        name='tensorboard',
        version=None
    )
    loggers.append(tb_logger)

    # Weights & Biases logger (if installed)
    try:
        wandb_logger = WandbLogger(
            project='rq-vae-recommender',
            name=os.path.basename(os.getcwd()),
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        loggers.append(wandb_logger)
    except ImportError:
        pass

    return loggers

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set seed
    set_seed(cfg.seed)

    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Initialize data module
    datamodule = RecommenderDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    # Get dataset properties
    num_users = datamodule.num_users
    num_items = datamodule.num_items
    print(f"\nDataset properties: {num_users} users, {num_items} items")

    # Initialize model
    if cfg.task.name == 'train_rqvae':
        model = RQVAE(cfg, num_users=num_users, num_items=num_items)
    elif cfg.task.name == 'train_decoder':
        model = RQVAEDecoder(cfg, num_users=num_users, num_items=num_items)
    else:
        raise ValueError(f"Unknown task: {cfg.task.name}")

    # Print model summary
    print(f"\nModel summary:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Configure callbacks and loggers
    callbacks = configure_callbacks(cfg)
    loggers = configure_loggers(cfg)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator='auto',
        devices='auto',
        strategy='auto',
        logger=loggers,
        callbacks=callbacks,
        enable_checkpointing=True,
        default_root_dir='./logs',
        gradient_clip_val=cfg.train.get('gradient_clip_val', 0.0),
        gradient_clip_algorithm=cfg.train.get('gradient_clip_algorithm', 'norm'),
        limit_train_batches=cfg.train.get('limit_train_batches', 1.0),
        limit_val_batches=cfg.train.get('limit_val_batches', 1.0),
        limit_test_batches=cfg.train.get('limit_test_batches', 1.0),
        fast_dev_run=cfg.train.get('fast_dev_run', False),
        overfit_batches=cfg.train.get('overfit_batches', 0.0),
        log_every_n_steps=cfg.train.get('log_every_n_steps', 50),
        check_val_every_n_epoch=cfg.train.get('check_val_every_n_epoch', 1),
        num_sanity_val_steps=cfg.train.get('num_sanity_val_steps', 2),
        precision=32,
    )

    # Train model
    if cfg.task.name in ['train_rqvae', 'train_decoder']:
        trainer.fit(model, datamodule=datamodule)

        # Test model
        if cfg.train.get('run_test', True):
            trainer.test(model, datamodule=datamodule, ckpt_path='best')

    # Evaluate only
    elif cfg.task.name == 'eval':
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.eval.checkpoint.path)

    # Prediction only
    elif cfg.task.name == 'predict':
        predictions = trainer.predict(model, datamodule=datamodule, ckpt_path=cfg.predict.checkpoint.path)
        torch.save(predictions, './predictions.pt')
        print(f"Predictions saved to ./predictions.pt")

if __name__ == "__main__":
    main()
