"""
Visualizes the pose for the commavq dataset
"""

from typing import Tuple

import cv2
import numpy as np


def convert_3D_points_to_2D(points_3D, homo_cam_mat):
    points_2D = []
    for index in range(points_3D.shape[0]):
        p4d = points_3D[index]
        p2d = (homo_cam_mat) @ p4d
        px, py = 0, 0
        if p2d[2][0] != 0.0:
            px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])

        points_2D.append([px, py])

    return np.array(points_2D)


def get_rect_coords(x_i, y_i, x_j, y_j, width=2.83972):
    Pi = np.array([x_i, y_i])
    Pj = np.array([x_j, y_j])
    height = np.linalg.norm(Pi - Pj)
    diagonal = (width**2 + height**2) ** 0.5
    D = diagonal / 2.0

    M = ((Pi + Pj) / 2.0).reshape((2,))
    theta = np.arctan2(Pi[1] - Pj[1], Pi[0] - Pj[0])
    theta += np.pi / 4.0
    points = np.array(
        [
            M
            + np.array(
                [
                    D * np.sin(theta + 0 * np.pi / 2.0),
                    D * np.cos(theta + 0 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 1 * np.pi / 2.0),
                    D * np.cos(theta + 1 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 2 * np.pi / 2.0),
                    D * np.cos(theta + 2 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 3 * np.pi / 2.0),
                    D * np.cos(theta + 3 * np.pi / 2.0),
                ]
            ),
        ]
    )
    return points


def get_rect_coords_3D(Pi, Pj, width=0.25):
    x_i, y_i = Pi[0, 0], Pi[2, 0]
    x_j, y_j = Pj[0, 0], Pj[2, 0]
    points_2D = get_rect_coords(x_i, y_i, x_j, y_j, width)
    points_3D = []
    for index in range(points_2D.shape[0]):
        # point_2D = points_2D[index]
        point_3D = Pi.copy()
        point_3D[0, 0] = points_2D[index, 0]
        point_3D[2, 0] = points_2D[index, 1]

        points_3D.append(point_3D)

    return np.array(points_3D)


def plot_steering_traj(
    frame_center,
    trajectory,
    color=(255, 0, 0),
    intrinsic_matrix=None,
    DistCoef=None,
    offsets=[0.0, -1.8, -1.5],
    method="add_weighted",
    track=False,
):
    assert method in ("overlay", "mask", "add_weighted")

    h, w = frame_center.shape[:2]

    if intrinsic_matrix is None:
        intrinsic_matrix = estimate_intrinsics(65.6, 42.4, h, w)
    if DistCoef is None:
        DistCoef = np.array(
            [
                0.0177,
                3.8938e-04,  # Tangential Distortion
                -0.1533,
                0.4539,
                -0.6398,  # Radial Distortion
            ]
        )
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        intrinsic_matrix, DistCoef, (w, h), 1, (w, h)
    )
    homo_cam_mat = np.hstack((intrinsic_matrix, np.zeros((3, 1))))

    # rot = trajectory[0][:3,:3]
    # rot = np.eye(3,3)
    prev_point = None
    prev_point_3D = None
    rect_frame = np.zeros_like(frame_center)

    for trajectory_point in trajectory:
        p4d = np.ones((4, 1))
        p3d = np.array(
            [
                trajectory_point[0] * 1 - offsets[0],
                # trajectory_point[1] * 1 - offsets[1],
                -offsets[1],
                trajectory_point[2] * 1 - offsets[2],
            ]
        ).reshape((3, 1))
        # p3d = np.linalg.inv(rot) @ p3d
        p4d[:3, :] = p3d

        p2d = (homo_cam_mat) @ p4d
        if (
            p2d[2][0] != 0.0
            and not np.isnan(p2d).any()
            and not np.isinf(p2d).any()
        ):
            px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])
            # frame_center = cv2.circle(frame_center, (px, py), 2, color, -1)
            if prev_point is not None:
                px_p, py_p = prev_point
                if track:
                    rect_coords_3D = get_rect_coords_3D(p4d, prev_point_3D)
                    rect_coords = convert_3D_points_to_2D(
                        rect_coords_3D, homo_cam_mat
                    )
                    rect_frame = cv2.fillPoly(
                        rect_frame, pts=[rect_coords], color=color
                    )

                frame_center = cv2.line(
                    frame_center, (px_p, py_p), (px, py), color, 2
                )

            prev_point = (px, py)
            prev_point_3D = p4d.copy()
        else:
            prev_point = None
            prev_point_3D = None

    if method == "mask":
        mask = np.logical_and(
            rect_frame[:, :, 0] == color[0],
            rect_frame[:, :, 1] == color[1],
            rect_frame[:, :, 2] == color[2],
        )
        frame_center[mask] = color
    elif method == "overlay":
        frame_center += (0.2 * rect_frame).astype(np.uint8)
    elif method == "add_weighted":
        cv2.addWeighted(frame_center, 1.0, rect_frame, 0.2, 0.0, frame_center)
    return frame_center


def plot_bev_trajectory(trajectory, frame_center, color=(0, 255, 0)):
    WIDTH, HEIGHT = frame_center.shape[1], frame_center.shape[0]
    traj_plot = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    Z = trajectory[:, 2]
    X = trajectory[:, 0]

    RAN = 20.0
    X_min, X_max = -RAN, RAN
    # Z_min, Z_max = -RAN, RAN
    Z_min, Z_max = -0.1 * RAN, RAN
    X = (X - X_min) / (X_max - X_min)
    Z = (Z - Z_min) / (Z_max - Z_min)

    # X = (X - lb) / (ub - lb)
    # Z = (Z - lb) / (ub - lb)

    for traj_index in range(1, X.shape[0]):
        u = int(
            np.round(np.clip((X[traj_index] * (WIDTH - 1)), -1, WIDTH + 1))
        )
        v = int(
            np.round(np.clip((Z[traj_index] * (HEIGHT - 1)), -1, HEIGHT + 1))
        )
        u_p = int(
            np.round(np.clip((X[traj_index - 1] * (WIDTH - 1)), -1, WIDTH + 1))
        )
        v_p = int(
            np.round(
                np.clip((Z[traj_index - 1] * (HEIGHT - 1)), -1, HEIGHT + 1)
            )
        )

        if u < 0 or u >= WIDTH or v < 0 or v >= HEIGHT:
            continue

        traj_plot = cv2.circle(traj_plot, (u, v), 2, color, -1)
        traj_plot = cv2.line(traj_plot, (u_p, v_p), (u, v), color, 2)

    traj_plot = cv2.flip(traj_plot, 0)
    return traj_plot


def estimate_intrinsics(
    fov_x: float,  # degrees
    fov_y: float,  # degrees
    height: int,  # pixels
    width: int,  # pixels
) -> np.ndarray:
    """
    The intrinsic matrix can be extimated from the FOV and image dimensions

    :param fov_x: FOV on x axis in degrees
    :type fov_x: float
    :param fov_y: FOV on y axis in degrees
    :type fov_y: float
    :param height: Height in pixels
    :type height: int
    :param width: Width in pixels
    :type width: int
    :returns: (3,3) intrinsic matrix
    """
    fov_x = np.deg2rad(fov_x)
    fov_y = np.deg2rad(fov_y)

    if fov_x == 0.0 or fov_y == 0.0:
        raise ZeroDivisionError("fov can't be zero")

    c_x = width / 2.0
    c_y = height / 2.0
    f_x = c_x / np.tan(fov_x / 2.0)
    f_y = c_y / np.tan(fov_y / 2.0)

    intrinsic_matrix = np.array(
        [
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1],
        ],
        dtype=np.float16,
    )

    return intrinsic_matrix


def visualize_frame_img(
    img: np.ndarray,
    trajectory: np.ndarray,
    color: Tuple[int, int, int],
):

    dx = trajectory[1:, 2] - trajectory[:-1, 2]
    speed = dx / (1.0 / 2.0)
    # m/s to km/h
    speed_kmph = speed * 3.6

    img = plot_steering_traj(
        img,
        trajectory,
        color=color,
    )

    img_bev = plot_bev_trajectory(trajectory, img, color=color)

    # Write speed on img
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2

    img = cv2.resize(img, (0, 0), fx=2, fy=2)
    img_bev = cv2.resize(img_bev, (0, 0), fx=2, fy=2)

    cv2.putText(
        img,
        f"Speed: {speed_kmph.mean():.2f} kmph",
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType,
    )

    return img, img_bev
