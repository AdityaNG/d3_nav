import argparse
import os

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

from d3nav.datasets.nusc import NuScenesDataset
from d3nav.model.trainer import D3NavTrajTrainingModule
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint"
    )
    args = parser.parse_args()

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

    # Load model
    model = D3NavTrajTrainingModule.load_from_checkpoint(args.ckpt)
    model.eval()

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

        # Get the last frame (most recent)
        last_frame = x[-1]  # Shape: (3, H, W)

        # Convert from normalized tensor back to image format
        frame = last_frame.transpose(1, 2, 0)  # Shape: (H, W, 3)
        frame = (frame).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Get reconstructed trajectory
        with torch.no_grad():
            y_tensor = torch.from_numpy(y).unsqueeze(0)  # Add batch dimension
            y_tensor = y_tensor.to(device="cuda")
            pred_trajectory = model(y_tensor)
            pred_trajectory = (
                pred_trajectory[0].cpu().numpy()
            )  # Remove batch dimension

        # Process original trajectory
        original_traj = y[:, [1, 2, 0]]
        original_traj[:, 0] *= -1
        original_traj = np.vstack(
            ([0, 0, 0], original_traj)
        )  # Add origin point

        # Process reconstructed trajectory
        recon_traj = pred_trajectory[:, [1, 2, 0]]
        recon_traj[:, 0] *= -1
        recon_traj = np.vstack(([0, 0, 0], recon_traj))  # Add origin point

        # Create two separate visualizations
        img_orig, img_bev_orig = visualize_frame_img(
            img=frame.copy(),
            trajectory=original_traj,
            color=(0, 255, 0),  # Green for original
        )

        img_recon, img_bev_recon = visualize_frame_img(
            img=frame.copy(),
            trajectory=recon_traj,
            color=(0, 0, 255),  # Red for reconstructed
        )

        # Stack original and reconstructed vertically
        combined_cam = np.vstack([img_orig, img_recon])
        combined_bev = np.vstack([img_bev_orig, img_bev_recon])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            combined_cam, "Original", (10, 30), font, 1, (255, 255, 255), 2
        )
        cv2.putText(
            combined_cam,
            "Reconstructed",
            (10, combined_cam.shape[0] // 2 + 30),
            font,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            combined_bev, "Original", (10, 30), font, 1, (255, 255, 255), 2
        )
        cv2.putText(
            combined_bev,
            "Reconstructed",
            (10, combined_bev.shape[0] // 2 + 30),
            font,
            1,
            (255, 255, 255),
            2,
        )

        # Combine camera view and BEV horizontally
        final_vis = np.hstack([combined_cam, combined_bev])

        # Save the visualization
        cv2.imwrite("vis/vis.png", final_vis)

        import time

        time.sleep(0.25)


if __name__ == "__main__":
    main()
