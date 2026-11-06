from __future__ import absolute_import, division, print_function

import logging

import torch
import torch.nn as nn

from src.models.bifpn import BiFPNLayer
from src.models.effecientnet import *

logger = logging.getLogger(__name__)

bn_norm_kwargs = {'momentum': 0.01, 'eps': 1e-4, 'affine': True}
gn_norm_kwargs = {'eps': 1e-4, 'affine': True}

'''
This modified EfficientDet now includes an additional decoder
to predict heatmaps from the BiFPN features.
'''


class EfficientDet(nn.Module):
    def __init__(self, cfg, scaled_parameters):
        super().__init__()

        phi = cfg.model.phi
        # Build EfficientNet backbone (using efficientnet_b{phi} as example)
        self.efficientnet = eval(f'efficientnet_b{phi}')(pretrained=cfg.model.pretrained)
        # Remove last Convolutional head (1x1), AvgPool and FC layers
        self.efficientnet = nn.ModuleList(self.efficientnet.children())[:-2][0][:-1]

        # For heatmap prediction, we optionally use P2 features.
        self.use_p2 = True

        efficientnet_out_channels = [
            [16, 24, 40, 112, 320],  # b0
            [16, 24, 40, 112, 320],  # b1
            [16, 24, 48, 120, 352],  # b2
            [24, 32, 48, 136, 384],  # b3
            [24, 32, 56, 160, 448],  # b4
            [24, 40, 64, 176, 512],  # b5
            [32, 40, 72, 200, 576],  # b6
            [32, 48, 80, 224, 640],  # b7

        ]
        # Build the BiFPN layers
        self.bifpn = nn.Sequential(
            *[
                BiFPNLayer(
                    efficientnet_out_channels[phi],
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
        features = [None]  # Placeholder (e.g. for P1 if needed)
        for idx in range(len(self.efficientnet)):
            x = self.efficientnet[idx](x)
            if idx == 2:
                features.append(x) if self.use_p2 else features.append(None)
            if idx in [3, 5, 7]:
                features.append(x)
        backbone_feat = features[-1] 
        bifpn_features = self.bifpn(features)

        return bifpn_features[1],backbone_feat

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
    Get all relevant scaled parameters to build EfficientPose.
    """
    # info tuples with scalable parameters
    # image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    bifpn_widths = (64, 88, 112, 160, 224, 288, 384,384)
    bifpn_depths = (3, 4, 5, 6, 7, 7, 8,8)
    subnet_depths = (3, 3, 3, 4, 4, 4, 5,5)
    subnet_iteration_steps = (1, 1, 1, 2, 2, 2, 3,3)
    num_groups_gn = (4, 4, 7, 10, 14, 18, 24,24)
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
    model = EfficientDet(cfg, get_scaled_parameters(cfg))

    return model

