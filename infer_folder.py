"""Run keypoint inference on a folder of raw images (no json / no ground truth).

Every image in `input_dir` is padded + resized exactly like the val/test pipeline,
pushed through the trained heatmap model, and written back to `vis_dir` with the
predicted keypoints drawn on top of the original (unpadded) image.

Usage:
    python infer_folder.py +input_dir=/path/to/pngs +vis_dir=/path/to/output \
        general.checkpoint_path=/path/to/model.ckpt

Optional overrides:
    +device=cuda:0          # default cuda:0 if available, else cpu
    +batch_size=8
    +conf_thr=0.01          # keypoints below this confidence are drawn faded
    +draw_skeleton=true     # connect the 8 box corners
    +save_json=true         # also dump predicted keypoints to <vis_dir>/predictions.json
"""

import glob
import json
import math
import os
import warnings

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.datasets.get_dataset import load_transforms
from src.tllib.metric import get_max_preds
from src.utils.hydra import load_obj, resolve_paths, set_padded_image_size

IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')

# The 8 keypoints describe a box: 0-3 top face, 4-7 bottom face (see points.json).
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # top face
    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

KP_COLORS = [
    (255, 56, 56), (255, 157, 51), (255, 231, 51), (150, 255, 51),
    (51, 255, 133), (51, 235, 255), (51, 133, 255), (204, 51, 255),
]


class FolderImageDataset(Dataset):
    """Reads plain images from a folder and prepares them like KeypointDataset(val/test)."""

    def __init__(self, files, cfg: DictConfig, transform):
        self.files = files
        self.transforms = transform
        self.input_image_size = tuple(cfg.datamodule.input_size)  # (w, h)
        self.padded_image_size = tuple(cfg.datamodule.padded_image_size)  # (w, h)

    def __len__(self):
        return len(self.files)

    def pad_to_aspect_ratio(self, image):
        """Same padding as the val/test dataset: pad only right/bottom, so keypoints in
        padded space are already in original-image space."""
        h, w = image.shape[:2]
        pad_w, pad_h = self.padded_image_size
        # If an image is bigger than the configured padded canvas, grow the canvas by
        # whole multiples of the input size so the geometry stays consistent.
        if w > pad_w or h > pad_h:
            k = math.ceil(max(w / self.input_image_size[0], h / self.input_image_size[1]))
            pad_w = max(pad_w, int(self.input_image_size[0] * k))
            pad_h = max(pad_h, int(self.input_image_size[1] * k))
        pad_right = int(pad_w - w)
        pad_bottom = int(pad_h - h)
        padded = cv2.copyMakeBorder(
            image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded

    def __getitem__(self, index):
        path = self.files[index]
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f'Could not read image: {path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        padded = self.pad_to_aspect_ratio(image)
        padded_h, padded_w = padded.shape[:2]
        img = cv2.resize(padded, self.input_image_size)

        t_image = self.transforms(image=img, keypoints=np.zeros((0, 2), dtype=np.float32))['image']
        # padded size is returned per-image so keypoints can be rescaled correctly.
        return t_image, path, padded_w, padded_h


def draw_predictions(image, keypoints, conf, conf_thr, draw_skeleton):
    """Draw keypoints (and optionally the box skeleton) on a BGR image."""
    vis = image.copy()
    h, w = vis.shape[:2]
    radius = max(2, int(round(0.004 * max(h, w))))
    thickness = max(1, int(round(0.002 * max(h, w))))
    font_scale = max(0.4, 0.0006 * max(h, w))

    pts = [(int(round(x)), int(round(y))) for x, y in keypoints]

    if draw_skeleton:
        for i, j in SKELETON:
            if i >= len(pts) or j >= len(pts):
                continue
            if conf[i] < conf_thr or conf[j] < conf_thr:
                continue
            cv2.line(vis, pts[i], pts[j], (60, 220, 60), thickness, cv2.LINE_AA)

    for k, (x, y) in enumerate(pts):
        color = KP_COLORS[k % len(KP_COLORS)]
        low = conf[k] < conf_thr
        cv2.circle(vis, (x, y), radius, color, -1 if not low else 1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f'{k}:{conf[k]:.2f}',
            (x + radius + 2, y - radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return vis


def run(cfg: DictConfig) -> None:
    input_dir = cfg.get('input_dir', None)
    if input_dir is None:
        raise ValueError('Please pass the image folder, e.g. `+input_dir=/path/to/images`')
    if cfg.general.checkpoint_path is None:
        raise ValueError('Please pass a checkpoint: `general.checkpoint_path=/path/to/model.ckpt`')
    # Hydra changes the working directory to the run directory: resolve relative paths first.
    input_dir = hydra.utils.to_absolute_path(input_dir)
    vis_dir = hydra.utils.to_absolute_path(cfg.get('vis_dir', os.path.join(os.getcwd(), 'predictions_vis')))
    resolve_paths(cfg)
    os.makedirs(vis_dir, exist_ok=True)

    device = cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = int(cfg.get('batch_size', cfg.training.batch_size))
    conf_thr = float(cfg.get('conf_thr', cfg.datamodule.confidence_threshold))
    draw_skeleton = bool(cfg.get('draw_skeleton', False))
    save_json = bool(cfg.get('save_json', True))

    files = sorted(
        f for ext in IMG_EXTS for f in glob.glob(os.path.join(input_dir, '**', ext), recursive=True)
    )
    if not files:
        raise ValueError(f'No images found in {input_dir}')
    print(f'Found {len(files)} images in {input_dir}')

    # Same padded canvas as train/test.
    set_padded_image_size(cfg)

    model = load_obj(cfg.training.lightningmodule).load_from_checkpoint(
        cfg.general.checkpoint_path,
        cfg=cfg,
        strict=True,
        map_location=torch.device(device),
    )
    model.eval().to(device)
    print(f'Loaded checkpoint {cfg.general.checkpoint_path} on {device}')

    dataset = FolderImageDataset(files, cfg, load_transforms(cfg.transforms.test))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
    )

    results = []
    with torch.no_grad():
        for images, paths, padded_ws, padded_hs in loader:
            images = images.to(device)
            pred_heatmaps, _, _ = model.model(images)
            pred_heatmaps = pred_heatmaps.detach().float().cpu().numpy()
            keypoints, conf = get_max_preds(pred_heatmaps)  # heatmap coordinates

            out_h, out_w = pred_heatmaps.shape[2:]
            for i, path in enumerate(paths):
                scale_w = float(padded_ws[i]) / out_w
                scale_h = float(padded_hs[i]) / out_h
                # Padding is applied only right/bottom, so padded coordinates are already
                # in original-image coordinates.
                kpts = keypoints[i] * np.array([scale_w, scale_h], dtype=np.float32)
                scores = conf[i].reshape(-1)

                orig = cv2.imread(path)
                vis = draw_predictions(orig, kpts, scores, conf_thr, draw_skeleton)

                rel = os.path.relpath(path, input_dir)
                out_path = os.path.join(vis_dir, os.path.splitext(rel)[0] + '_kpts.png')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, vis)

                results.append(
                    {
                        'filename': rel,
                        'keypoints': kpts.tolist(),
                        'confidence': scores.tolist(),
                    }
                )
            print(f'{len(results)}/{len(files)} images done', end='\r')

    print(f'\nSaved {len(results)} visualizations to {vis_dir}')
    if save_json:
        json_path = os.path.join(vis_dir, 'predictions.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved predicted keypoints to {json_path}')


@hydra.main(version_base='1.1', config_path='conf', config_name='eclipse_cfg')
def run_model(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    warnings.filterwarnings('ignore')
    torch.set_float32_matmul_precision('medium')
    torch.multiprocessing.set_sharing_strategy('file_system')
    run_model()
