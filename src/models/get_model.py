from omegaconf import DictConfig
from src.models.pose_efficientnet import get_pose_net
from src.models.pose_pvt_v2 import get_pose_net as get_pvt_v2_pose_net

class ModelPoseEffecientNet:
    def __init__(self, cfg: DictConfig):
        self.model = get_pose_net(cfg)

class ModelPosePvtV2:
    def __init__(self, cfg: DictConfig):
        self.model = get_pvt_v2_pose_net(cfg)

