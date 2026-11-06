# ECLIPSE Baselines

Official baseline code for **ECLIPSE: A Spacecraft Rendezvous Trajectories Dataset with Controlled
In-Orbit Lighting Conditions** (NeurIPS 2026, Track on Evaluations and Datasets).

[Project page](https://chenni-nidhaleddine.github.io/ECLIPSE/) ·
[Dataset](https://huggingface.co/datasets/CVI2-UniLU/ECLIPSE) ·
[Pretrained weights](https://drive.google.com/drive/folders/1xR6VZ4ky1Ven1QN-mV2eOO5wDzNhwYLS?usp=sharing)

ECLIPSE pairs synthetic training images with real images of a spacecraft mock-up captured along
14 rendezvous trajectories, each under 4 lighting conditions (L01–L04). This repository provides
the baselines for monocular 6D pose estimation on it.

The baseline is a keypoint-heatmap regressor:

1. **Backbone + neck:** an EfficientNet-B*φ* backbone (ImageNet-pretrained) with a BiFPN neck.
   A PVTv2 backbone is also available.
2. **Head:** a decoder predicts one heatmap per 3D keypoint. ECLIPSE uses 8 keypoints, the
   corners of the spacecraft bounding box.
3. **Pose:** the most confident keypoints (5 by default) are lifted to a 6D pose with PnP +
   RANSAC and refined with Levenberg–Marquardt.

## Installation

```bash
git clone https://github.com/CHENNI-Nidhaleddine/ECLIPSE_baselines.git
cd ECLIPSE_baselines

conda create -n eclipse python=3.11 -y
conda activate eclipse
# Install the PyTorch build that matches your CUDA version, e.g. CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Dataset

Download ECLIPSE from [Hugging Face](https://huggingface.co/datasets/CVI2-UniLU/ECLIPSE) into a
folder named `eclipse_final` inside a data root (here `./data`), then extract the per-trajectory
archives:

```bash
huggingface-cli download CVI2-UniLU/ECLIPSE --repo-type dataset --local-dir data/eclipse_final
cd data/eclipse_final
for z in synthetic_images/*.zip; do unzip -q "$z" -d synthetic_images && rm "$z"; done
for z in real_images/*.zip;      do unzip -q "$z" -d real_images      && rm "$z"; done
```

The code expects this layout:

```
data/
└── eclipse_final/
    ├── synthetic_images/        # T000/T000_000.png, ...
    ├── real_images/             # T01L01/T01L01_000.png, ...
    ├── train_syn.json           # synthetic training split
    ├── validation_syn.json      # synthetic validation split
    ├── test_real.json           # all real images
    ├── L01.json ... L04.json    # real images, one lighting condition each
    ├── points.json              # 3D keypoints of the spacecraft model
    └── real_camera.json         # camera intrinsics and distortion
```

Each annotation file is a list of records:

```json
{
  "filename": "T000/T000_000.png",
  "keypoints": [x0, y0, x1, y1, ...],     // 2D projections of the keypoints, in pixels
  "r_xyz": [tx, ty, tz],                  // translation, in meters
  "q_xyzw": [qx, qy, qz, qw]              // orientation quaternion
}
```

The data root defaults to `./data`. To use another location, pass `general.dataset_dir=/path/to/data`.

## Training

```bash
# EfficientNet-B5 (default), trained on synthetic images and validated on the synthetic validation split
python train.py general.dataset_dir=/path/to/data

# EfficientNet-B3, 2 GPUs, custom run name
python train.py model.phi=3 trainer.devices=2 general.id=effb3

# PVTv2-B0 backbone
python train.py model=pose_pvt_v2 training=pose_pvt_v2
```

Each run writes to `outputs/<general.id>/<date_time>/`:

- `saved_models/`: Lightning checkpoints (best by validation loss, best by keypoint accuracy, last),
  plus a plain PyTorch `.pth` file with the best weights.
- `logs/`: TensorBoard and W&B logs. W&B runs offline by default; set `general.wandb_offline=False`
  to sync, or `logging.log=False` to turn logging off.
- `code/`: a snapshot of the code used for the run.

## Evaluation

```bash
# On all real images
python test.py general.checkpoint_path=/path/to/model.ckpt datamodule.group=target

# On one lighting condition
python test.py general.checkpoint_path=/path/to/model.ckpt datamodule.group=target \
    'datamodule.test_data_json=${general.dataset_root}/L04.json'

# On the synthetic validation split
python test.py general.checkpoint_path=/path/to/model.ckpt datamodule.group=source
```

If the checkpoint filename contains `=` (for example `epoch=25-acc_avg=0.94.ckpt`), quote it so
Hydra reads it correctly: `"general.checkpoint_path='/path/to/epoch=25-acc_avg=0.94.ckpt'"`.

The script prints the following metrics (mean and std) and the number of rejected poses:

| Metric | Description |
|---|---|
| `test_ang_err` | Rotation error, in degrees |
| `test_loc_err_norm` | Translation error divided by the ground-truth distance |
| `test_speed_score` | Pose score: rotation error (rad) + normalized translation error |
| `test_accuracy` | Fraction of keypoints within 15 px of the ground truth |

The per-image predictions are saved to `test_predictions_<general.id>.json` in the run directory.

## Inference on a folder of images

Predict the keypoints for every image in a folder (no annotations needed) and save the
visualizations:

```bash
python infer_folder.py general.checkpoint_path=/path/to/model.ckpt \
    +input_dir=/path/to/images +vis_dir=/path/to/output \
    +draw_skeleton=true
```

## Configuration

All settings are [Hydra](https://hydra.cc) configs in [conf/](conf), and any value can be
overridden from the command line.

| Group | File | Content |
|---|---|---|
| `general` | `conf/general/eclipse.yaml` | Paths, run name, seed, checkpoint |
| `datamodule` | `conf/datamodule/keypoint_eclipse.yaml` | Splits, image/heatmap sizes, number of keypoints, PnP settings |
| `model` | `conf/model/pose_efficientnet.yaml` | Backbone (`phi`) and ImageNet initialization |
| `training` | `conf/training/pose_efficientnet.yaml` | Batch size, workers, monitored metrics |
| `trainer` | `conf/trainer/server.yaml` | PyTorch Lightning `Trainer` arguments (GPUs, epochs, precision) |
| `transforms` | `conf/transforms/default.yaml` | Albumentations photometric augmentations |
| `optimizer`, `scheduler` | `conf/optimizer/opt.yaml`, `conf/scheduler/default.yaml` | AdamW and MultiStepLR |

Use `general.debug=True` to run on the first 100 samples of each split.

## Repository structure

```
├── train.py                 # training
├── test.py                  # evaluation (keypoints + 6D pose)
├── infer_folder.py          # inference and visualization on raw images
├── conf/                    # Hydra configs
├── src/
│   ├── datasets/            # keypoint dataset, padding/resizing, heatmap targets
│   ├── lightning/           # LightningModules and DataModule
│   ├── models/              # EfficientNet, PVTv2, BiFPN, heatmap head
│   ├── tllib/               # heatmap decoding, keypoint accuracy, PnP and pose metrics
│   └── utils/               # Hydra helpers
└── tests/                   # unit tests: python -m unittest tests/test_augmentation.py
```

## Citation

If you use this code or the ECLIPSE dataset, please cite:

```bibtex
@inproceedings{chenni2026eclipse,
  title     = {{ECLIPSE}: A Spacecraft Rendezvous Trajectories Dataset
               with Controlled In-Orbit Lighting Conditions},
  author    = {Chenni, Nidhal Eddine and Rathinam, Arunkumar and
               Ali, Abid and Aouada, Djamila},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Track on Evaluations and Datasets},
  year      = {2026}
}
```

## Acknowledgements

This code builds on:

- [SPNv2](https://github.com/tpark94/spnv2) (BiFPN layers, MIT License)
- [torchvision](https://github.com/pytorch/vision) (EfficientNet implementation, BSD-3-Clause License)
- [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) (heatmap utilities, MIT License)

## License

The code is released under the [MIT License](LICENSE). Files adapted from other projects keep
their original license notice (see the header of each file). The dataset has its own license; see
the [dataset page](https://huggingface.co/datasets/CVI2-UniLU/ECLIPSE).
