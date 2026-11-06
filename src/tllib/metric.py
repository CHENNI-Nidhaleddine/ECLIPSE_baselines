"""
Source: https://github.com/thuml/Transfer-Learning-Library
License: thuml/Transfer-Learning-Library is licensed under the MIT License
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""

from typing import Optional

import numpy as np


def refine_keypoints(heatmaps, keypoints):
    """
    Refines keypoints using bilinear interpolation.
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    keypoints: numpy.ndarray([batch_size, num_joints, 2]) -> (x, y)
    """
    batch_size, num_joints, h, w = heatmaps.shape
    refined_keypoints = keypoints.copy()
    # Sub-pixel refinement with a second-order Taylor expansion (as in DARK).
    for b in range(batch_size):
        for j in range(num_joints):
            x, y = keypoints[b, j].astype(int)
            if 1 <= x < w - 1 and 1 <= y < h - 1:  # Ensure valid 3x3 region
                patch = heatmaps[b, j, y - 1 : y + 2, x - 1 : x + 2]  # Extract 3x3 patch

                if patch.sum() > 1e-6:  # Avoid division by zero
                    # Compute gradients in x and y directions
                    dx = 0.5 * (patch[1, 2] - patch[1, 0])
                    dy = 0.5 * (patch[2, 1] - patch[0, 1])

                    # Compute second-order gradients
                    dxx = patch[1, 2] + patch[1, 0] - 2 * patch[1, 1]
                    dyy = patch[2, 1] + patch[0, 1] - 2 * patch[1, 1]
                    dxy = 0.25 * (patch[2, 2] + patch[0, 0] - patch[0, 2] - patch[2, 0])

                    # Solve for subpixel offset using Taylor expansion
                    hessian = np.array([[dxx, dxy], [dxy, dyy]])
                    grad = np.array([-dx, -dy])

                    if np.linalg.det(hessian) > 1e-6:  # Ensure invertibility
                        subpixel_shift = np.linalg.inv(hessian) @ grad
                        refined_keypoints[b, j] += subpixel_shift  # Adjust keypoints

    return refined_keypoints


def get_max_preds(batch_heatmaps):
    """
    Get predictions from heatmaps using argmax + refinement
    """
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_heatmaps should be 4D'

    batch_size, num_joints, height, width = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, axis=2)
    maxvals = np.amax(heatmaps_reshaped, axis=2).reshape((batch_size, num_joints, 1))

    preds = np.zeros((batch_size, num_joints, 2), dtype=np.float32)
    preds[:, :, 0] = idx % width
    preds[:, :, 1] = idx // width

    pred_mask = (maxvals > 0).astype(np.float32)
    preds *= pred_mask  # Mask invalid predictions

    # Apply refinement
    refined_preds = refine_keypoints(batch_heatmaps, preds)

    return refined_preds, maxvals


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    '''Return percentage below threshold while ignoring values with a -1'''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy_dist(output, target, hm_type='gaussian'):
    """
    Calculate accuracy in terms of distance,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    """
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        norm = np.ones((pred.shape[0], 2))
    dists = calc_dists(pred, target, norm)

    return dists.T


def calc_dists_fixed(preds, target):
    """
    Calculate raw pixel distances without normalization.
    preds: [B, num_joints, 2]
    target: [B, num_joints, 2]
    returns: [num_joints, B] distance matrix
    """
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))

    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = np.linalg.norm(preds[n, c] - target[n, c])
            else:
                # Keypoint outside the image: ignored by dist_acc.
                dists[c, n] = -1
    return dists


def accuracy(output, target, padded_image_size, hm_type='gaussian', pixel_threshold=15):
    """
    Calculates accuracy based on a fixed pixel threshold (e.g. 15px on 1920x1200 images)
    """
    idx = list(range(output.shape[1]))
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)

        # Scale keypoints to original image size
        output_width = output.shape[3]
        output_height = output.shape[2]

        # Heatmap coordinates -> padded image coordinates
        padded_image_width, padded_image_height = padded_image_size
        scale_x = padded_image_width / output_width
        scale_y = padded_image_height / output_height

        pred_scaled = pred.copy()
        target_scaled = target.copy()

        pred_scaled[..., 0] *= scale_x
        pred_scaled[..., 1] *= scale_y
        target_scaled[..., 0] *= scale_x
        target_scaled[..., 1] *= scale_y

        dists = calc_dists_fixed(pred_scaled, target_scaled)

    acc = np.zeros(len(idx))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i] = dist_acc(dists[idx[i]], thr=pixel_threshold)
        if acc[i] >= 0:
            avg_acc += acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    return acc, avg_acc, cnt, pred_scaled


class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
