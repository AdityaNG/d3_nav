import os

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

from d3nav.datasets.nusc import NuScenesDataset
from d3nav.visual import visualize_frame_img


def get_camera_intrinsic(nusc, camera_channel="CAM_FRONT"):
    # Get first scene
    first_scene = nusc.scene[0]

    # Get first sample in the scene
    sample = nusc.get("sample", first_scene["first_sample_token"])

    # Get camera sample data
    cam_front_data = nusc.get("sample_data", sample["data"][camera_channel])

    # Get calibrated sensor data
    calib_sensor = nusc.get(
        "calibrated_sensor", cam_front_data["calibrated_sensor_token"]
    )

    # Get camera intrinsic matrix
    camera_intrinsic = np.array(calib_sensor["camera_intrinsic"])

    return camera_intrinsic


def main():
    # Create output directory if it doesn't exist
    os.makedirs("vis", exist_ok=True)

    # Initialize NuScenes
    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot="/media/NG/datasets/nuscenes/",
        verbose=True,
    )

    camera_intrinsic = get_camera_intrinsic(nusc)
    print("Camera Intrinsic Matrix:")
    print(camera_intrinsic)

    # Create dataset
    dataset = NuScenesDataset(
        nusc=nusc,
        is_train=False,  # Use validation set for visualization
        prediction_horizon=6,
        fps=2,
    )

    # Process one frame at a time
    for idx in range(len(dataset)):
        print(f"Processing frame {idx}/{len(dataset)}")

        # Get data for this frame
        x, y, _ = dataset[idx]  # x: frames, y: trajectory

        print("y", y.shape)
        # Get the last frame (most recent)
        last_frame = x[-1]  # Shape: (3, H, W)

        # Convert from normalized tensor back to image format
        frame = last_frame.transpose(1, 2, 0)  # Shape: (H, W, 3)
        frame = (frame).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        trajectory = y[:, [1, 2, 0]]
        trajectory[:, 0] *= -1
        trajectory = np.vstack(
            ([0, 0, 0], trajectory)
        )  # Add origin point at start

        # Visualize the frame with trajectory
        img_vis, img_bev = visualize_frame_img(
            img=frame,
            trajectory=trajectory,
            color=(0, 255, 0),  # Green color for trajectory
        )

        # Combine the visualizations horizontally
        combined_vis = np.hstack([img_vis, img_bev])

        # Save the visualization
        cv2.imwrite("vis/vis.png", combined_vis)

        import time

        time.sleep(0.25)


if __name__ == "__main__":
    main()
