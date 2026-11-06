"""Evaluate a trained checkpoint (keypoint accuracy + 6D pose errors).

Usage:
    python test.py general.checkpoint_path=/path/to/model.ckpt general.dataset_dir=/path/to/data
    # evaluate on the real test split instead of the synthetic validation split:
    python test.py ... datamodule.group=target
"""
import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.utils.hydra import load_obj, resolve_paths, set_padded_image_size, set_seed


def run(cfg: DictConfig) -> None:
    set_seed(cfg.training.seed)
    cfg.general.mode = 'test'

    if cfg.general.checkpoint_path is None:
        raise ValueError('Please pass a checkpoint: `general.checkpoint_path=/path/to/model.ckpt`')
    resolve_paths(cfg)
    ckpt_file_path = cfg.general.checkpoint_path

    set_padded_image_size(cfg)
    dm = load_obj(cfg.training.datamodule)(cfg=cfg)

    loggers = []
    if cfg.logging.log:
        for logger in cfg.logging.loggers:
            loggers.append(load_obj(logger.class_name)(**logger.params))

    trainer = pl.Trainer(logger=loggers, **cfg.trainer)

    model = load_obj(cfg.training.lightningmodule).load_from_checkpoint(
        ckpt_file_path,
        cfg=cfg,
        strict=True,
        map_location='cpu',
    )
    trainer.test(model=model, datamodule=dm)


@hydra.main(version_base='1.1', config_path='conf', config_name='eclipse_cfg')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    run(cfg)


if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    warnings.filterwarnings('ignore')
    torch.set_float32_matmul_precision('medium')
    torch.multiprocessing.set_sharing_strategy('file_system')
    run_model()
