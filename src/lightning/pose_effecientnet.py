from omegaconf import DictConfig

from src.lightning.pose_base import LitPoseBase
from src.models.get_model import ModelPoseEffecientNet


class LitPoseEffecientNet(LitPoseBase):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _build_model(self, cfg: DictConfig):
        self.pose_effecientnet = ModelPoseEffecientNet(cfg)
        return self.pose_effecientnet.model
