import importlib
from itertools import product
from typing import Any, Dict, Generator
import os
import random
import shutil

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning.utilities import rank_zero_only


def load_obj(obj_path: str, default_obj_path: str = '') -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path

    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)


def product_dict(**kwargs: Dict) -> Generator:
    """
    Convert dict with lists in values into lists of all combinations

    This is necessary to convert config with experiment values
    into format usable by hydra
    Args:
        **kwargs:

    Returns:
        list of lists

    ---
    Example:
        >>> list_dict = {'a': [1, 2], 'b': [2, 3]}
        >>> list(product_dict(**list_dict))
        >>> [['a=1', 'b=2'], ['a=1', 'b=3'], ['a=2', 'b=2'], ['a=2', 'b=3']]

    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        zip_list = list(zip(keys, instance))
        yield [f'{i}={j}' for i, j in zip_list]


def config_to_hydra_dict(cfg: DictConfig) -> Dict:
    """
    Convert config into dict with lists of values, where key is full name of parameter

    This fuction is used to get key names which can be used in hydra.

    Args:
        cfg:

    Returns:
        converted dict

    """
    experiment_dict = {}
    for k, v in cfg.items():
        for k1, v1 in v.items():
            experiment_dict[f'{k}.{k1}'] = v1

    return experiment_dict


def resolve_paths(cfg: DictConfig) -> None:
    """Make user-provided paths absolute.

    Hydra changes the working directory to the run directory (`general.output_dir`) before the
    task function is called, so relative paths given on the command line are resolved against
    the directory the script was launched from.
    """
    with open_dict(cfg):
        cfg.general.output_dir = os.getcwd()
        cfg.general.dataset_dir = hydra.utils.to_absolute_path(cfg.general.dataset_dir)
        if cfg.general.checkpoint_path is not None:
            cfg.general.checkpoint_path = hydra.utils.to_absolute_path(cfg.general.checkpoint_path)


def set_padded_image_size(cfg: DictConfig) -> None:
    """Fill `cfg.datamodule.padded_image_size` (left as null in the config files).

    Images are zero-padded (randomly during training, right/bottom at val/test time) to a
    canvas of `input_size * padding_scale_factor` before being resized to `input_size`.
    """
    factor = cfg.datamodule.padding_scale_factor
    input_size = cfg.datamodule.input_size
    image_size = cfg.datamodule.image_size
    padded = (int(input_size[0] * factor), int(input_size[1] * factor))
    if padded[0] < image_size[0] or padded[1] < image_size[1]:
        raise ValueError(
            f"Padded canvas {padded} is smaller than the image size {tuple(image_size)}; "
            f"increase datamodule.padding_scale_factor."
        )
    with open_dict(cfg):
        cfg.datamodule.padded_image_size = padded


def set_seed(seed: int = 666) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@rank_zero_only
def save_useful_info() -> None:
    """Snapshot the code (src/ and train.py) into `<run dir>/code` for reproducibility."""
    code_dir = os.path.join(os.getcwd(), 'code')
    shutil.copytree(
        os.path.join(REPO_ROOT, 'src'),
        os.path.join(code_dir, 'src'),
        ignore=shutil.ignore_patterns('__pycache__'),
        dirs_exist_ok=True,
    )
    shutil.copy2(os.path.join(REPO_ROOT, 'train.py'), code_dir)
