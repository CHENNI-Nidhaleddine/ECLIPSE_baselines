"""
Easy sanity checks for the geometric-augmentation pipeline used by
`KeypointDataset` (src/datasets/PoseEffecientNetDataset.py).

The core property under test: whenever albumentations warps the image, the
keypoints returned alongside it must still land on the same physical content
(a keypoint drawn on someone's shoulder must still be on that shoulder after
a flip/rotate/shift-scale). We verify this generically -- for *any*
geometric transform -- by drawing distinct-colored circular markers at each
keypoint location and checking that, after the transform, each keypoint
coordinate still sits on top of its matching marker color.

Running `test_visualize_augmentations` also dumps before/after PNGs to
test/output/ so the result can be checked visually, not just numerically.

Run with:
    python -m unittest test/test_augmentation.py -v
"""
import os
import unittest

import albumentations as A
import cv2
import numpy as np
from albumentations.core.composition import Compose, KeypointParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Mirrors conf/datamodule/keypoint_eclipse.yaml num_keypoints.
NUM_KEYPOINTS = 8
IMAGE_SIZE = 256
MARKER_RADIUS = 8
# Keypoints are placed on a ring of this radius around the image center, so
# that after scale (~1.15x) and rotation they still stay inside the frame
# and away from border-fill artifacts.
RING_RADIUS = 50

# One visually distinct color per keypoint (order matters: index i always
# maps to color i), so a mixed-up ordering after a transform is also caught.
KEYPOINT_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230),
]


def make_synthetic_sample(size=IMAGE_SIZE, num_keypoints=NUM_KEYPOINTS, ring_radius=RING_RADIUS):
    """Build a synthetic RGB image with a distinct colored marker at each keypoint."""
    image = np.full((size, size, 3), 40, dtype=np.uint8)  # dark gray background
    center = np.array([size / 2, size / 2])
    angles = np.linspace(0, 2 * np.pi, num_keypoints, endpoint=False)
    keypoints = np.stack(
        [center[0] + ring_radius * np.cos(angles), center[1] + ring_radius * np.sin(angles)],
        axis=1,
    ).astype(np.float32)
    for (x, y), color in zip(keypoints, KEYPOINT_COLORS):
        cv2.circle(image, (int(round(x)), int(round(y))), MARKER_RADIUS, color, -1)
    return image, keypoints


def color_at(image, xy, half_win=2):
    """Average color in a small patch around (x, y), clamped to image bounds."""
    x, y = xy
    h, w = image.shape[:2]
    xi = int(round(np.clip(x, 0, w - 1)))
    yi = int(round(np.clip(y, 0, h - 1)))
    patch = image[max(0, yi - half_win):yi + half_win + 1, max(0, xi - half_win):xi + half_win + 1]
    return patch.reshape(-1, 3).mean(axis=0)


def assert_keypoints_track_markers(test_case, image, keypoints, colors=KEYPOINT_COLORS, tol=60, msg=""):
    """Each keypoint must still sit on top of its matching colored marker."""
    test_case.assertEqual(len(keypoints), len(colors))
    for i, (kp, expected_color) in enumerate(zip(keypoints, colors)):
        actual_color = color_at(image, kp)
        diff = np.abs(actual_color - np.array(expected_color)).max()
        test_case.assertLessEqual(
            diff,
            tol,
            f"{msg} keypoint {i} at {tuple(kp)} is off its marker "
            f"(expected color {expected_color}, got {tuple(actual_color.round(1))})",
        )


def draw_keypoints(image, keypoints):
    vis = image.copy()
    for i, (x, y) in enumerate(keypoints):
        center = (int(round(x)), int(round(y)))
        cv2.drawMarker(vis, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
        cv2.putText(vis, str(i), (center[0] + 6, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def save_side_by_side(path, left, right, left_title, right_title):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    top_pad = 24
    gap = 10
    h = max(left.shape[0], right.shape[0]) + top_pad
    w = left.shape[1] + gap + right.shape[1]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[top_pad:top_pad + left.shape[0], :left.shape[1]] = left
    canvas[top_pad:top_pad + right.shape[0], left.shape[1] + gap:left.shape[1] + gap + right.shape[1]] = right
    cv2.putText(canvas, left_title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, right_title, (left.shape[1] + gap + 5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def build_transform(augs):
    """Same Compose construction as src/datasets/get_dataset.py:load_transforms."""
    return Compose(augs, keypoint_params=KeypointParams(format='xy', remove_invisible=False))


class TestGeometricAugmentation(unittest.TestCase):
    def setUp(self):
        self.image, self.keypoints = make_synthetic_sample()

    def test_synthetic_sample_sanity(self):
        # Sanity-check the test fixture itself before trusting it to test anything else.
        assert_keypoints_track_markers(self, self.image, self.keypoints, msg="[fixture]")

    def test_horizontal_flip_keeps_keypoints_on_markers(self):
        out = build_transform([A.HorizontalFlip(p=1.0)])(image=self.image, keypoints=self.keypoints)
        aug_kpts = np.array(out['keypoints'])
        self.assertEqual(len(aug_kpts), NUM_KEYPOINTS)
        assert_keypoints_track_markers(self, out['image'], aug_kpts, msg="[hflip]")

    def test_vertical_flip_keeps_keypoints_on_markers(self):
        out = build_transform([A.VerticalFlip(p=1.0)])(image=self.image, keypoints=self.keypoints)
        assert_keypoints_track_markers(self, out['image'], np.array(out['keypoints']), msg="[vflip]")

    def test_rotate_keeps_keypoints_on_markers(self):
        out = build_transform(
            [A.Rotate(limit=(35, 35), border_mode=cv2.BORDER_CONSTANT, p=1.0)]
        )(image=self.image, keypoints=self.keypoints)
        assert_keypoints_track_markers(self, out['image'], np.array(out['keypoints']), msg="[rotate35]")

    def test_affine_shift_scale_rotate_keeps_keypoints_on_markers(self):
        out = build_transform([
            A.Affine(
                scale=1.1, translate_percent=0.08, rotate=20,
                border_mode=cv2.BORDER_CONSTANT, p=1.0,
            )
        ])(image=self.image, keypoints=self.keypoints)
        assert_keypoints_track_markers(self, out['image'], np.array(out['keypoints']), msg="[affine]")

    def test_visualize_augmentations(self):
        """Dumps before/after PNGs under test/output/ for manual visual inspection."""
        augmentations = {
            "hflip": A.HorizontalFlip(p=1.0),
            "vflip": A.VerticalFlip(p=1.0),
            "rotate35": A.Rotate(limit=(35, 35), border_mode=cv2.BORDER_CONSTANT, p=1.0),
            "affine_shift_scale_rotate": A.Affine(
                scale=1.1, translate_percent=0.08, rotate=20,
                border_mode=cv2.BORDER_CONSTANT, p=1.0,
            ),
        }
        original_vis = draw_keypoints(self.image, self.keypoints)
        for name, aug in augmentations.items():
            out = build_transform([aug])(image=self.image, keypoints=self.keypoints)
            aug_vis = draw_keypoints(out['image'], np.array(out['keypoints']))
            path = os.path.join(OUTPUT_DIR, f"aug_{name}.png")
            save_side_by_side(path, original_vis, aug_vis, "original", name)
            self.assertTrue(os.path.exists(path))
        print(f"\nSaved visual-check images to {OUTPUT_DIR}")


if __name__ == '__main__':
    unittest.main()
