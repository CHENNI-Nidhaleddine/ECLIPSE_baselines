import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger

from src.tllib.metric import AverageMeter, accuracy, get_max_preds
from src.tllib.pose import (
    PoseEstimator,
    error_orientation,
    error_translation,
    speed_score,
)
from src.utils.hydra import load_obj

logger = logging.getLogger(__name__)


def to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def _to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class LitPoseBase(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.image_size = self.cfg.datamodule.image_size
        self.padded_image_size = self.cfg.datamodule.padded_image_size
        self.model = self._build_model(cfg)
        self.outputs = []
        self.rejs = 0
        self.automatic_optimization = True
        self.results = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()
        self.accuracy = AverageMeter("Acc (s)", ":3.2f")
        self.pbd = True  # progress bar display
        camera_json = cfg.datamodule.camera_json
        points_json = cfg.datamodule.points_json
        with open(camera_json, 'r') as file:
            self.camera_params = json.load(file)
        with open(points_json, 'r') as file:
            self.points = json.load(file)

        self.pose_estimator = self._build_pose_estimator()
        self.test_metrics = []
        self.train_img_path = self.cfg.datamodule.train_img_path
        self.train_data_json = self.cfg.datamodule.train_data_json

        self.num_points_for_pose = self.cfg.datamodule.num_points_for_pose

    # --- Hooks implemented by subclasses ---------------------------------

    def _build_model(self, cfg: DictConfig):
        raise NotImplementedError

    def _build_pose_estimator(self) -> PoseEstimator:
        return PoseEstimator(
            self.points['keypoints'],
            self.camera_params['cameraMatrix'],
            self.camera_params['distCoeffs'],
        )

    def _select_pose_indices(self, conf_scores, filename):
        indices = list(range(len(conf_scores)))
        if len(indices) >= self.num_points_for_pose:  # Make sure we have required valid points
            indices.sort(key=lambda x: conf_scores[x], reverse=True)
            return indices[: self.num_points_for_pose]
        print(f"Not enough points for {filename}")
        return None

    # --- Shared Lightning logic --------------------------------------------

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = load_obj(self.cfg.optimizer.finetune.class_name)(
            params, **self.cfg.optimizer.finetune.params
        )
        lr_sch = {
            "scheduler": load_obj(self.cfg.scheduler.class_name)(
                optimizer, **self.cfg.scheduler.params
            ),
            "interval": self.cfg.scheduler.interval,
            "frequency": 1,
            "monitor": self.cfg.training.metric_val_loss,
        }
        return [optimizer], [lr_sch]

    def compute_loss_heatmaps(self, pred_heatmaps, tgt_heatmaps):
        # Add small epsilon to avoid numerical instability
        epsilon = 1e-10

        # Normalize predictions and target heatmaps
        pred_heatmaps = pred_heatmaps + epsilon
        tgt_heatmaps = tgt_heatmaps + epsilon
        pred_heatmaps = pred_heatmaps / pred_heatmaps.sum(dim=(2, 3), keepdim=True)
        tgt_heatmaps = tgt_heatmaps / tgt_heatmaps.sum(dim=(2, 3), keepdim=True)

        # Compute KL divergence loss
        loss = F.kl_div(pred_heatmaps.log(), tgt_heatmaps, reduction='batchmean')
        return loss

    def training_step(self, batch, batch_idx, *args, **kwargs):
        images, tgt_heatmaps = batch[0], batch[1]
        pred_heatmaps, backbone_feats,bifpn_feats = self.model(images)
        task_loss = self.compute_loss_heatmaps(pred_heatmaps, tgt_heatmaps)
        _, avg_acc, cnt, _ = accuracy(
            to_numpy(pred_heatmaps),
            to_numpy(tgt_heatmaps),
            self.padded_image_size,
        )
        self.log_dict(
            {"train_acc": avg_acc, "loss task": task_loss}, prog_bar=True, sync_dist=True
        )

        self.log(
            "epoch_loss",
            task_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            logger=False,
        )
        return task_loss

    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        epoch_loss = self.trainer.callback_metrics.get("epoch_loss")
        if epoch_loss is None:
            return
        for pl_logger in self.loggers:
            if isinstance(pl_logger, TensorBoardLogger):
                pl_logger.experiment.add_scalar(
                    "epoch_loss", epoch_loss.item(), self.current_epoch
                )

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        images, tgt_heatmaps = batch[0], batch[1]
        pred_heatmaps, backbone_feats, bifpn_feats = self.model(images)
        loss = self.compute_loss_heatmaps(pred_heatmaps, tgt_heatmaps)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        _, avg_acc, cnt, pred = accuracy(
            to_numpy(pred_heatmaps),
            to_numpy(tgt_heatmaps),
            self.padded_image_size,
        )

        self.accuracy.update(avg_acc, cnt)

        return loss

    def on_validation_epoch_end(self):
        sum_ = torch.tensor(self.accuracy.sum, device=self.device, dtype=torch.float)
        count_ = torch.tensor(self.accuracy.count, device=self.device, dtype=torch.float)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_, op=dist.ReduceOp.SUM)
        global_avg = (sum_ / count_).item()

        msg = '\n Epoch: {0} \t Accuracy ({acc:.3f}) \n'.format(self.current_epoch, acc=global_avg)
        logger.info(msg)
        self.log_dict({"acc_avg": global_avg}, prog_bar=self.pbd, sync_dist=False)
        self.accuracy.reset()
        return

    def test_step(self, batch, batch_idx):
        images, gt_heatmaps, meta = batch[0], batch[1], batch[2]
        pred_heatmaps, backbone_feats,bifpn_feats = self.model(images)
        pred_keypoints, pred_conf = get_max_preds(to_numpy(pred_heatmaps))
        gt_keypoints, _ = get_max_preds(to_numpy(gt_heatmaps))

        acc, avg_acc, cnt, pred = accuracy(
            to_numpy(pred_heatmaps),
            to_numpy(gt_heatmaps),
            self.padded_image_size,
        )
        self.accuracy.update(avg_acc, cnt)
        output_height, output_width = pred_heatmaps.shape[2:]
        padded_image_width, padded_image_height = self.padded_image_size
        scale_height = padded_image_height / output_height
        scale_width = padded_image_width / output_width
        assert scale_height == scale_width, "Scale factors for height and width should be equal."

        pred_keypoints = pred_keypoints * scale_height
        gt_keypoints = gt_keypoints * scale_height

        for i in range(len(images)):
            try:
                # Safely access meta data
                device = images[i].device
                filename = (
                    meta['filename'][i] if isinstance(meta['filename'], list) else meta['filename']
                )
                t_gt = torch.tensor(
                    [meta['r_xyz'][0][i], meta['r_xyz'][1][i], meta['r_xyz'][2][i]]
                ).to(device)
                q_gt = torch.tensor(
                    [
                        meta['q_xyzw'][0][i],
                        meta['q_xyzw'][1][i],
                        meta['q_xyzw'][2][i],
                        meta['q_xyzw'][3][i],
                    ]
                ).to(device)

                keypoints = pred_keypoints[i]
                keypoints_gt = gt_keypoints[i]
                conf_scores = pred_conf[i]

                indices = self._select_pose_indices(conf_scores, filename)
                if indices is None:
                    continue
                q_est, t_est = self.pose_estimator.solve_pose2(
                    keypoints, np.array(indices), RANSAC=True
                )
                
                reject = False
                if self.cfg.general.dataset_id == 'speedplus':
                    if t_est[2] < 0 or t_est[2] > 50:
                        reject = True
                        self.rejs = self.rejs + 1
                    else:
                        # --- 2. Check xy limits based on estimated z distance
                        x_max = t_est[2] * 0.326 #* np.tan(camera["horizontalFOV"]/2)
                        y_max = x_max / self.camera_params["Nu"] * self.camera_params["Nv"]
                    
                        if t_est[0] < -x_max or t_est[0] > x_max or t_est[1] < -y_max or t_est[1] > y_max:
                            reject = True
                            self.rejs=self.rejs+1
                if self.cfg.general.dataset_id == 'eclipse_final':
                    if t_est[2] < 0 or t_est[2] > 50:
                        reject = True
                        self.rejs = self.rejs + 1

                # Calculate metrics
                t_gt = t_gt.detach().cpu().numpy()
                q_gt = q_gt.detach().cpu().numpy()
                loc_err = error_translation(t_est, t_gt)
                t_gt2 = np.reshape(t_gt, (3,))
                loc_err_norm = loc_err / np.sqrt(np.sum(np.square(t_gt2)))

                ang_err = error_orientation(q_est, q_gt)
                speed = speed_score(
                    t_est, q_est, t_gt, q_gt, rotThresh=0.0, posThresh=0.000  # in degrees
                )
                if not reject:
                    self.test_metrics.append(
                        {
                            'loc_err': float(loc_err),
                            'loc_err_norm': float(loc_err_norm),
                            'ang_err': float(ang_err),
                            'speed_score': float(speed),
                            "filename": filename,
                            'accuracy': self.accuracy.val,
                        }
                    )
                    self.outputs.append(
                        {
                            "t_est": list(t_est),
                            "q_est": list(q_est),
                            "keypoints": list(keypoints),
                            "keypoints_gt": list(keypoints_gt),
                            "filename": filename,
                            'accuracy': self.accuracy.val,
                        }
                    )

            except Exception as e:
                print(f"Error processing {filename if 'filename' in locals() else 'unknown file'}")
                print(f"Error details: {str(e)}")
                continue
        if self.test_metrics:
            metrics_df = pd.DataFrame(self.test_metrics)
            avg_metrics = {
                'test_loc_err_norm': metrics_df['loc_err_norm'].mean(),
                'test_loc_err_norm_std': metrics_df['loc_err_norm'].std(),
                'test_ang_err': metrics_df['ang_err'].mean(),
                'test_ang_err_std': metrics_df['ang_err'].std(),
                'test_speed_score': metrics_df['speed_score'].mean(),
                'test_speed_score_std': metrics_df['speed_score'].std(),
                'test_accuracy': metrics_df['accuracy'].mean(),
                'test_accuracy_std': metrics_df['accuracy'].std(),
            }
            print(f"[rank {self.global_rank}] {avg_metrics}")

    def on_test_epoch_end(self):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_outputs = [None] * world_size
            dist.all_gather_object(gathered_outputs, self.outputs)
            all_outputs = [item for rank_outputs in gathered_outputs for item in rank_outputs]

            gathered_test_metrics = [None] * world_size
            dist.all_gather_object(gathered_test_metrics, self.test_metrics)
            all_test_metrics = [
                item for rank_metrics in gathered_test_metrics for item in rank_metrics
            ]
        else:
            all_outputs = self.outputs
            all_test_metrics = self.test_metrics

        if all_test_metrics and self.trainer.is_global_zero:
            metrics_df = pd.DataFrame(all_test_metrics)
            avg_metrics = {
                'test_loc_err_norm': metrics_df['loc_err_norm'].mean(),
                'test_loc_err_norm_std': metrics_df['loc_err_norm'].std(),
                'test_ang_err': metrics_df['ang_err'].mean(),
                'test_ang_err_std': metrics_df['ang_err'].std(),
                'test_speed_score': metrics_df['speed_score'].mean(),
                'test_speed_score_std': metrics_df['speed_score'].std(),
                'test_accuracy': metrics_df['accuracy'].mean(),
                'test_accuracy_std': metrics_df['accuracy'].std(),
            }
            print('Global avg ', len(metrics_df['loc_err_norm']), avg_metrics)
            print('Number of rejections ', self.rejs)

        if all_outputs and self.trainer.is_global_zero:
            output_path = os.path.join(
                self.cfg.general.output_dir, f'test_predictions_{self.cfg.general.id}.json'
            )
            with open(output_path, 'w') as f:
                json.dump(all_outputs, f, default=_to_json_serializable, indent=2)
            logger.info(f"Saved {len(all_outputs)} test predictions to {output_path}")
        return
