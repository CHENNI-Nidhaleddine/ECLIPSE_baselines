import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from kornia.geometry.subpix import dsnt
from omegaconf import DictConfig
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose


class KeypointDataset(Dataset):
    """A generic dataset class for image keypoint detection"""

    def __init__(self, cfg: DictConfig, transform: Compose):
        self.cfg = cfg
        self.transforms = transform
        self.mode = cfg.general.mode
        self.group = cfg.datamodule.group

        self.dm = self.cfg.datamodule
        self.train_img_path = self.cfg.datamodule.train_img_path
        self.train_data_json = self.cfg.datamodule.train_data_json
        self.val_img_path = self.cfg.datamodule.val_img_path
        self.val_data_json = self.cfg.datamodule.val_data_json
        self.test_img_path = self.cfg.datamodule.test_img_path
        self.test_data_json = self.cfg.datamodule.test_data_json

        self.heatmap_size = np.array(self.dm.output_size, dtype=int)
        self.input_image_size = self.dm.input_size
        self.num_keypoints = int(self.dm.num_keypoints)
        std = torch.Tensor(self.dm.heatmap_sigma)
        self.heatmap_sigma = std.repeat(self.num_keypoints, 1)
        self.input_image_size = self.dm.input_size
        self.padded_image_size = self.dm.padded_image_size
        self.num_keypoints = int(self.dm.num_keypoints)
        # Pick the annotation file and the matching image folder for the current split.
        #   train            -> synthetic training split
        #   val/test, source -> synthetic validation split
        #   val/test, target -> real test split
        if self.mode == 'train':
            data_json, self.img_path = self.train_data_json, self.train_img_path
        elif self.mode in ['val', 'test']:
            if self.group == 'source':
                data_json, self.img_path = self.val_data_json, self.val_img_path
            elif self.group == 'target':
                data_json, self.img_path = self.test_data_json, self.test_img_path
            else:
                raise ValueError(f"datamodule.group must be 'source' or 'target', got {self.group}")
        else:
            raise ValueError(f"general.mode must be 'train', 'val' or 'test', got {self.mode}")
        self.df = pd.read_json(data_json)

        if self.cfg.general.debug:
            self.df = self.df[:100]

    def edit_keypoints(self, kpts, num_keypoints):
        kpts = np.array(kpts).reshape(num_keypoints, -1)
        vi = np.ones(num_keypoints)
        kpts = kpts[:, 0:2]
        return kpts, vi

    def pad_to_aspect_ratio_train(self, image, keypoints):
        """
        Pad image to target aspect ratio (input_w / input_h),
        shifting keypoints accordingly.
        """
        # generic formula
        img_size = image.shape[1], image.shape[0]  # (width/cols, height/rows)
        diff_h = self.padded_image_size[1] - img_size[1]
        diff_w = self.padded_image_size[0] - img_size[0]

        pad_top =random.randint(0, diff_h)
        pad_bot = int(diff_h - pad_top)
        pad_left =random.randint(0, diff_w)
        pad_right =int(diff_w - pad_left)

        image_padded = cv2.copyMakeBorder(
            image, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        keypoints_padded = keypoints + np.array([pad_left, pad_top], dtype=np.float32)
        return image_padded, keypoints_padded, pad_left, pad_right, pad_top, pad_bot

    def pad_to_aspect_ratio_val(self, image, keypoints):
        """
        Pad image to target aspect ratio (input_w / input_h),
        shifting keypoints accordingly.
        """
        h, w = image.shape[:2]

        # generic formula
        img_size = image.shape[1], image.shape[0]  # (width/cols, height/rows)
        diff_h = self.padded_image_size[1] - img_size[1]
        diff_w = self.padded_image_size[0] - img_size[0]
        # Deterministic padding (right/bottom only) so that predictions map back to image coordinates.
        pad_top = 0
        pad_bot = diff_h - pad_top
        pad_left = 0
        pad_right = diff_w - pad_left
        pad_bot = int(pad_bot)
        pad_right = int(pad_right)

        image_padded = cv2.copyMakeBorder(
            image, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        keypoints_padded = keypoints + np.array([pad_left, pad_top], dtype=np.float32)
        return image_padded, keypoints_padded, pad_left, pad_top

    def __getitem__(self, index):
        idata = self.df.iloc[index]
        fname = idata['filename']

        if self.mode in ['train']:
            image = cv2.imread(os.path.join(self.img_path, fname))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            kpts, visible = self.edit_keypoints(idata['keypoints'], self.num_keypoints)
            padded_image, kpts, _, _, _, _ = self.pad_to_aspect_ratio_train(image, kpts)
            h, w, c = padded_image.shape
            img = cv2.resize(padded_image, self.input_image_size)

            # bring keypoints from padded-image space into the resized (transform-input) space
            resize_scale = np.array(
                [self.input_image_size[0] / w, self.input_image_size[1] / h], dtype=np.float32
            )
            kpts = kpts * resize_scale

            transformed = self.transforms(image=img, keypoints=kpts)
            t_image = transformed['image']
            kpts = np.array(transformed['keypoints'], dtype=np.float32).reshape(
                self.num_keypoints, 2
            )

            scale = self.input_image_size[0] / self.dm.output_size[0]

            keypoints = torch.tensor(kpts / scale, dtype=torch.float32)
            visible[
                np.logical_and(
                    np.logical_and(kpts[:, 0] > 0, kpts[:, 0] <= self.input_image_size[0]),
                    np.logical_and(kpts[:, 1] > 0, kpts[:, 1] <= self.input_image_size[1]),
                )
            ] = 1

            visibility = torch.tensor(visible).float()

            target = dsnt.render_gaussian2d(
                mean=keypoints,
                std=self.heatmap_sigma,
                size=tuple([self.heatmap_size[1], self.heatmap_size[0]]),
                normalized_coordinates=False,
            )
            target = target / (target.sum(dim=(1, 2), keepdim=True) + 1e-10)

            # target = target * visibility.view(len(visible), 1, 1)
            meta = {
                'filename': fname,
                'keypoints': keypoints,
                'image_size': (w, h),
                'visibility': visibility,
            }
            return t_image, target

        if self.mode in ['val', 'test']:
            image = cv2.imread(os.path.join(self.img_path, fname))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            kpts, visible = self.edit_keypoints(idata['keypoints'], self.num_keypoints)
            padded_image, kpts, _, _ = self.pad_to_aspect_ratio_val(image, kpts)
            h, w, c = padded_image.shape
            img = cv2.resize(padded_image, self.input_image_size)

            resize_scale = np.array(
                [self.input_image_size[0] / w, self.input_image_size[1] / h], dtype=np.float32
            )
            kpts = kpts * resize_scale

            transformed = self.transforms(image=img, keypoints=kpts)
            t_image = transformed['image']
            kpts = np.array(transformed['keypoints'], dtype=np.float32).reshape(
                self.num_keypoints, 2
            )

            scale = self.input_image_size[0] / self.dm.output_size[0]
            
            keypoints = torch.tensor(kpts / scale, dtype=torch.float32)
            visible[
                np.logical_and(
                    np.logical_and(kpts[:, 0] > 0, kpts[:, 0] <= self.input_image_size[0]),
                    np.logical_and(kpts[:, 1] > 0, kpts[:, 1] <= self.input_image_size[1]),
                )
            ] = 1

            visibility = torch.tensor(visible).float()
            target = dsnt.render_gaussian2d(
                mean=keypoints,
                std=self.heatmap_sigma,
                size=tuple([self.heatmap_size[1], self.heatmap_size[0]]),
                normalized_coordinates=False,
            )
            target = target / (target.sum(dim=(1, 2), keepdim=True) + 1e-10)
            # target = target * visibility.view(len(visible), 1, 1)
            meta = {
                'filename': fname,
                'keypoints': keypoints,
                'image_size': (w, h),
                'visibility': visibility,
                'r_xyz': idata['r_xyz'],
                'q_xyzw': idata['q_xyzw'],
            }
            return t_image, target, meta

    def __len__(self):
        return len(self.df)
