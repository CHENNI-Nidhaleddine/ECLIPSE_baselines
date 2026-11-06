import copy

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.datasets.get_dataset import load_transforms
from src.utils.hydra import load_obj


class KeypointDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        train_augs = load_transforms(self.cfg.transforms.train)
        train_source = load_obj(self.cfg.datamodule.class_name)(cfg=self.cfg, transform=train_augs)
        train_loader = DataLoader(
            train_source,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=True,
        )

        return train_loader

    def val_dataloader(self):
        val_augs = load_transforms(self.cfg.transforms.test)
        # Use a copy so that switching to 'val' mode does not affect the train/test datasets.
        val_cfg = copy.deepcopy(self.cfg)
        val_cfg.general.mode = 'val'
        val_source = load_obj(self.cfg.datamodule.class_name)(cfg=val_cfg, transform=val_augs)
        val_loader = DataLoader(
            val_source,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
        )
        return val_loader

    def test_dataloader(self):
        test_augs = load_transforms(self.cfg.transforms.test)
        test_target = load_obj(self.cfg.datamodule.class_name)(cfg=self.cfg, transform=test_augs)
        test_target_loader = DataLoader(
            test_target,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
        )
        return test_target_loader
