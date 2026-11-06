from __future__ import absolute_import, division, print_function

import logging

import torch
import torch.nn as nn

from src.models.bifpn import BiFPNLayer
from src.models.pvt_v2 import *

logger = logging.getLogger(__name__)

bn_norm_kwargs = {'momentum': 0.01, 'eps': 1e-4, 'affine': True}
gn_norm_kwargs = {'eps': 1e-4, 'affine': True}

'''
PVTv2 backbone + BiFPN neck, predicting heatmaps from the BiFPN features.
Mirrors EfficientDet in pose_efficientnet.py, but replaces the EfficientNet
backbone with PVTv2, whose 4 stages already give the P2~P5 pyramid levels
BiFPN expects (no manual feature tapping needed, unlike the conv backbone).
'''


class PvtV2Det(nn.Module):
    def __init__(self, cfg, scaled_parameters):
        super().__init__()

        phi = cfg.model.phi
        # Build PVTv2 backbone (using pvt_v2_b{phi} as example)
        self.pvt_v2 = eval(f'pvt_v2_b{phi}')(pretrained=cfg.model.pretrained)

        # For heatmap prediction, we optionally use P2 features.
        self.use_p2 = True

        pvt_v2_out_channels = [
            [32, 32, 64, 160, 256],  # b0
            [64, 64, 128, 320, 512],  # b1
            [64, 64, 128, 320, 512],  # b2
            [64, 64, 128, 320, 512],  # b3
            [64, 64, 128, 320, 512],  # b4
            [64, 64, 128, 320, 512],  # b5
        ]
        # Build the BiFPN layers
        self.bifpn = nn.Sequential(
            *[
                BiFPNLayer(
                    pvt_v2_out_channels[phi],
                    scaled_parameters['bifpn_width'],
                    first_time=True if i == 0 else False,
                    attention=True,
                    use_p2=self.use_p2,
                    use_group_norm=False,
                    group_norm_size=5,
                )
                for i in range(scaled_parameters['bifpn_depth'])
            ]
        )

        num_heatmaps = scaled_parameters.get('num_heatmaps', 1)
        self.heatmap_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                scaled_parameters['bifpn_width'],
                scaled_parameters['bifpn_width'] // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(scaled_parameters['bifpn_width'] // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                scaled_parameters['bifpn_width'] // 2,
                scaled_parameters['bifpn_width'] // 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(scaled_parameters['bifpn_width'] // 4),
            nn.ReLU(),
            nn.Conv2d(
                scaled_parameters['bifpn_width'] // 4,
                num_heatmaps,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),  # Ensures output is in [0,1]
        )

    def copy_weights(self, src_module, target_module):
        target_module.load_state_dict(src_module.state_dict())

    def forward_backbone(self, x):
        # outputs.hidden_states is already [None, P2, P3, P4, P5], matching
        # the feature format BiFPNLayer expects.
        outputs = self.pvt_v2(x, output_hidden_states=True)
        backbone_feat = outputs.hidden_states[-1]
        bifpn_features = self.bifpn(outputs.hidden_states)

        return bifpn_features[1], backbone_feat

    def forward_head(self, x):
        inter = self.heatmap_decoder[:6](x)
        heatmaps = self.heatmap_decoder[6:](inter)
        return heatmaps

    def forward(self, src):
        bifpn_features, backbone_feat = self.forward_backbone(src)
        pred_heatmaps = self.forward_head(bifpn_features)
        return pred_heatmaps, backbone_feat, bifpn_features


def get_scaled_parameters(cfg):
    """
    Get all relevant scaled parameters to build PvtV2Det.
    """
    # info tuples with scalable parameters
    # image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    bifpn_widths = (64, 88, 112, 160, 224, 288, 384)
    bifpn_depths = (3, 4, 5, 6, 7, 7, 8)
    subnet_depths = (3, 3, 3, 4, 4, 4, 5)
    subnet_iteration_steps = (1, 1, 1, 2, 2, 2, 3)
    num_groups_gn = (4, 4, 7, 10, 14, 18, 24)
    phi = cfg.model.phi
    num_keypoints = cfg.datamodule.num_keypoints
    parameters = {
        "bifpn_width": bifpn_widths[phi],
        "bifpn_depth": bifpn_depths[phi],
        "subnet_depth": subnet_depths[phi],
        "subnet_num_iteration_steps": subnet_iteration_steps[phi],
        "num_groups_gn": num_groups_gn[phi],
        "num_heatmaps": num_keypoints,  # Change this if you have more than 1 heatmap
    }
    return parameters


def get_pose_net(cfg):
    model = PvtV2Det(cfg, get_scaled_parameters(cfg))

    return model

