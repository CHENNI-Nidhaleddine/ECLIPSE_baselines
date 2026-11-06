import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def error_translation(t_pr, t_gt):
    t_pr = np.reshape(t_pr, (3,))
    t_gt = np.reshape(t_gt, (3,))
    return np.sqrt(np.sum(np.square(t_gt - t_pr)))


def error_orientation(q_pr, q_gt):
    # q must be [qvec, qcos]
    q_pr = np.reshape(q_pr, (4,))
    q_gt = np.reshape(q_gt, (4,))

    qdot = np.abs(np.dot(q_pr, q_gt))
    qdot = np.minimum(qdot, 1.0)
    return np.rad2deg(2 * np.arccos(qdot))  # [deg]

def quaternion_to_euler(w, x, y, z):
    # Roll (x-axis rotation)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    # Pitch (y-axis rotation)
    sin_pitch = 2 * (w * y - z * x)
    sin_pitch = np.clip(sin_pitch, -1, 1)  # Clamp to avoid numerical issues
    pitch = np.arcsin(sin_pitch)
    
    # Yaw (z-axis rotation)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Convert radians to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    return roll_deg, pitch_deg, yaw_deg


def speed_score(t_pr, q_pr, t_gt, q_gt, applyThresh=True, rotThresh=0.5, posThresh=0.005):
    # rotThresh: rotation threshold [deg]
    # posThresh: normalized translation threshold [m/m]
    t_gt=t_gt
    t_pr=t_pr
    err_t = error_translation(t_pr, t_gt)
    err_q = error_orientation(q_pr, q_gt)  # [deg]

    t_gt = np.reshape(t_gt, (3,))
    speed_t = err_t / np.sqrt(np.sum(np.square(t_gt)))
    speed_q = np.deg2rad(err_q)
 
    # Check if within threshold
    if applyThresh and err_q < rotThresh:
        speed_q = 0.0

    if applyThresh and speed_t < posThresh:
        speed_t = 0.0

    speed = speed_t + speed_q



    return speed


class PoseEstimator:
    def __init__(self, model_coordinates, camera_matrix,distCoeffs):

        self.model_points = np.array(model_coordinates).astype('float32')  # 3D model points.
        self.camera_matrix = np.array(camera_matrix).astype('float32')  # Camera intrinsics
        self.dist_coefs = np.array(distCoeffs, dtype=np.float32)  # Assuming no lens distortion
        # Rotation vector and translation vector
        self.r_vec = None
        self.t_vec = None

    def solve_pose(self, image_points, array, RANSAC=True, flag=0):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector and inliers) as output.
        """
        if RANSAC:
            # rotation and translation of board relative to camera
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.model_points[array],
                image_points[array],
                self.camera_matrix,
                self.dist_coefs,
                flags=flag)

        else:
            inliers, rvec, tvec = cv2.solvePnP(
                self.model_points[array],
                image_points[array],
                self.camera_matrix,
                self.dist_coefs,
                flags=flag)

        rmat, _ = cv2.Rodrigues(rvec)
        q = R.from_matrix(rmat).as_quat()
        r = tvec.squeeze()

        return q, r
    def solve_pose2(self, image_points, array, RANSAC, flag=0):
        """
        Solve pose from image points
        Return (rotation_vector and translation_vector) as output.
        """
        if (True):
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.model_points[array],
                image_points[array],
                self.camera_matrix,
                self.dist_coefs,
                flags=0)

        rvec, tvec = cv2.solvePnPRefineLM(
                self.model_points[array],
                image_points[array],
                self.camera_matrix,
                self.dist_coefs,
                rvec=rvec, tvec=tvec)


        rmat, _ = cv2.Rodrigues(rvec)
        rot_m = R.from_matrix(rmat)
        q = rot_m.as_quat()  # (x, y, z, w) format.
        r = tvec.squeeze()
        return q, r