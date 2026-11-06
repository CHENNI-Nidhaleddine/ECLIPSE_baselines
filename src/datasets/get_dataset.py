import omegaconf
from albumentations.core.composition import Compose, KeypointParams
from omegaconf import DictConfig

from src.utils.hydra import load_obj


def load_transforms(cfg: DictConfig) -> Compose:
    """
    Load albumentations

    Args:
        cfg:

    Returns:
        compose object
    """
    augs = []
    for a in cfg['augs']:
        if a['params'] is None:
            params = {}
        else:
            params = {k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v)) for k, v in
                      a['params'].items()}
        aug = load_obj(a['class_name'])(**params)
        augs.append(aug)

    # remove_invisible=False keeps a fixed-size, order-preserving keypoints array
    # so out-of-frame keypoints stay aligned with visibility/heatmap construction downstream.
    return Compose(augs, keypoint_params=KeypointParams(format='xy', remove_invisible=False))
