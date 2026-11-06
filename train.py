"""Train a heatmap-based keypoint regressor on the ECLIPSE synthetic images.

Usage:
    python train.py general.dataset_dir=/path/to/data
"""
import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from src.utils.hydra import load_obj, resolve_paths, save_useful_info, set_padded_image_size, set_seed


def run(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model

    Args:
        cfg: hydra config
    """
    if cfg.general.checkpoint_path is not None:
        raise ValueError(
            f"general.checkpoint_path={cfg.general.checkpoint_path} is only supported for "
            "inference (test.py / infer_folder.py)."
        )

    set_seed(cfg.training.seed)
    resolve_paths(cfg)
    set_padded_image_size(cfg)

    callbacks = [
        EarlyStopping(**cfg.callbacks.early_stopping.params),
        ModelCheckpoint(**cfg.callbacks.model_checkpoint.params),
        ModelCheckpoint(**cfg.callbacks.model_checkpoint_acc.params),
    ]

    for callback in cfg.callbacks.other_callbacks:
        if callback.params:
            callback_instance = load_obj(callback.class_name)(**callback.params)
        else:
            callback_instance = load_obj(callback.class_name)()
        callbacks.append(callback_instance)

    loggers = []
    if cfg.logging.log:
        for logger in cfg.logging.loggers:
            loggers.append(load_obj(logger.class_name)(**logger.params))

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        **cfg.trainer,
    )

    model = load_obj(cfg.training.lightningmodule)(cfg=cfg)
    dm = load_obj(cfg.training.datamodule)(cfg=cfg)
    trainer.fit(model, dm)

    _save_best_model(cfg, trainer, model)


@rank_zero_only
def _save_best_model(cfg: DictConfig, trainer: pl.Trainer, model: pl.LightningModule) -> None:
    """Additionally export the network weights (without Lightning state) as a plain .pth file."""
    if not cfg.general.save_pytorch_model:
        return
    os.makedirs('saved_models', exist_ok=True)

    if cfg.general.save_best:
        best_path = trainer.checkpoint_callback.best_model_path
        save_name = os.path.basename(os.path.normpath(best_path))
        model_class = load_obj(cfg.training.lightningmodule)
        weights = model_class.load_from_checkpoint(best_path, cfg=cfg, strict=False)
        model_name = f'saved_models/best_{save_name}'.replace('.ckpt', '.pth')
        torch.save(weights.model.state_dict(), model_name)
    else:
        torch.save(model.model.state_dict(), 'saved_models/last.pth')


@rank_zero_only
def _setup_logging(cfg: DictConfig) -> None:
    print(cfg.training)
    os.makedirs('logs', exist_ok=True)
    if cfg.general.log_code:
        save_useful_info()


@hydra.main(version_base='1.1', config_path='conf', config_name='eclipse_cfg')
def run_model(cfg: DictConfig) -> None:
    _setup_logging(cfg)
    run(cfg)


if __name__ == '__main__':
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    warnings.filterwarnings('ignore')
    torch.set_float32_matmul_precision('medium')
    torch.multiprocessing.set_sharing_strategy('file_system')
    run_model()
