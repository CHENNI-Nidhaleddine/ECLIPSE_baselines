from omegaconf import DictConfig

from src.lightning.pose_base import LitPoseBase
from src.models.get_model import ModelPosePvtV2


class LitPosePvtV2(LitPoseBase):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _build_model(self, cfg: DictConfig):
        self.pose_pvt_v2 = ModelPosePvtV2(cfg)
        return self.pose_pvt_v2.model
