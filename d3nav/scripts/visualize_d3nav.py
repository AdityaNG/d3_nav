import argparse
import math
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from d3nav.datasets.nusc import NuScenesDataset
from d3nav.model.trainer import D3NavTrajTrainingModule
from d3nav.scripts.train import temporal_context
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


@torch.no_grad()
def main():
    # ckpt = 'checkpoints/d3nav/d3nav-epoch-06-val_loss-0.6668.ckpt'  # 2 layers, starts from traj_quantizer/d3nav-traj-epoch-132-val_loss-0.2792.ckpt  # noqa
    # Gives crooked trajectories, probably because the GPT has to fit into the planner's embedding space and might have found a suboptimal representation  # noqa

    # ckpt = 'checkpoints/d3nav/d3nav-epoch-01-val_loss-0.6826.ckpt'  # 3 layers, starts from d3nav/d3nav-epoch-06-val_loss-0.6668.ckpt  # noqa
    # Increasing unfrozen GPT layers does not help  # noqa

    # ckpt = 'checkpoints/d3nav/d3nav-epoch-01-val_loss-0.5547.ckpt'  # 2 layers, traj unfrozen, starts from d3nav/d3nav-epoch-06-val_loss-0.6668.ckpt  # noqa
    # unfreezing the trajectory decoder unlocks a lot of performance. Model becomes very good at estimating speed, but many turns get missed, plus trajectory is still ever so slightly crooked  # noqa

    # ckpt = "checkpoints/d3nav/d3nav-epoch-03-val_loss-0.5128.ckpt"  # 3 layers, traj unfrozen, starts from d3nav/d3nav-epoch-06-val_loss-0.6668.ckpt  # noqa
    # Good checkpoint, l2 (1s) of 0.43, but had some train-val leakage  # noqa

    # ckpt = "checkpoints/d3nav/d3nav-epoch-03-val_loss-0.7735.ckpt"  # 3 layers, traj unfrozen, from scratch  # noqa
    # Good checkpoint, l2 (1s) of 0.70895, no train-val leakage  # noqa

    # ckpt = "checkpoints/d3nav/d3nav-epoch-15-val_loss-0.7631.ckpt"  # 3 layers, avg pool, cumulative, traj unfrozen, from scratch  # noqa
    # Learns turning, l2 (1s) of 0.67041

    # ckpt = "checkpoints/d3nav/d3nav-epoch-15-val_loss-0.6955.ckpt"  # 3 layers, ChunkedAttention, cumulative, traj unfrozen, from scratch  # noqa
    # Good turning, improves upon metrics

    # ckpt = "checkpoints/d3nav/d3nav-epoch-10-val_loss-0.9685.ckpt"  # Dropout 0.2, 3 layers, ChunkedAttention, cumulative, traj unfrozen, from scratch  # noqa
    # Frame level Dropout rate was too agressive

    # ckpt = "checkpoints/d3nav/d3nav-epoch-21-val_loss-1.5874.ckpt"  # 2 Frames only, random trajectory decoder, 3 layers, ChunkedAttention, cumulative, traj unfrozen, from scratch  # noqa
    # Gets to L1 1s of 1.0, but trajectories look a little wonky

    # ckpt = "checkpoints/d3nav/d3nav-epoch-06-val_loss-0.7179.ckpt"  # 2 Frames only, pretrained traj decoder, 3 layers, ChunkedAttention, cumulative, traj unfrozen, from scratch,   # noqa
    # Gets to L1 1s of 0.75, trajectories are smoother

    ckpt = "checkpoints/d3nav/d3nav-epoch-07-val_loss-1.2325.ckpt"  # 1 Frame only, pretrained traj decoder, 3 layers, ChunkedAttention, cumulative, traj unfrozen, from scratch,   # noqa
    # Gets to L1 1s of 1.41

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default=ckpt, help="Path to checkpoint"
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
    model.model.T = temporal_context
    model.eval()

    # from ..factory import load_d3nav
    # model = load_d3nav(args.ckpt)
    # model.eval()

    # Create dataset
    dataset = NuScenesDataset(
        nusc=nusc,
        is_train=False,  # Use validation set for visualization
        prediction_horizon=6,
        fps=2,
        temporal_context=temporal_context,
    )

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"vis/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Calculate video chunks
    fps = 2  # Dataset fps
    chunk_duration = 30  # seconds
    frames_per_chunk = fps * chunk_duration
    num_chunks = math.ceil(len(dataset) / frames_per_chunk)

    video_writer = None

    # Process frames in chunks
    for chunk in tqdm(range(num_chunks), desc="chunk"):
        print(f"Processing chunk {chunk+1}/{num_chunks}")

        # Calculate frame range for this chunk
        start_idx = chunk * frames_per_chunk
        end_idx = min((chunk + 1) * frames_per_chunk, len(dataset))
        # Process one frame at a time
        for idx in tqdm(range(start_idx, end_idx), desc="frame"):

            # Get data for this frame
            x, y, _ = dataset[idx]

            # Get the last frame (most recent)
            last_frame = x[-1]  # Shape: (3, H, W)

            # Convert from normalized tensor back to image format
            frame = last_frame.transpose(1, 2, 0)  # Shape: (H, W, 3)
            frame = (frame).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Get trajectories using both methods
            # 1. Direct trajectory prediction
            y_tensor = (
                torch.from_numpy(y).unsqueeze(0).to("cuda")
            )  # Add batch dimension
            pred_trajectory = model(y_tensor)
            pred_trajectory = (
                pred_trajectory[0].cpu().numpy()
            )  # Remove batch dimension

            # 2. Process image through D3Nav model
            processed_img = (
                torch.from_numpy(x).unsqueeze(0).to("cuda")
            )  # Add batch dimension
            img_trajectory = model.model(processed_img)
            img_trajectory = (
                img_trajectory[0].cpu().numpy()
            )  # Remove batch dimension

            # Process original trajectory
            original_traj = y[:, [1, 2, 0]]
            original_traj[:, 0] *= -1
            original_traj = np.vstack(
                ([0, 0, 0], original_traj)
            )  # Add origin point

            # Process direct prediction trajectory
            recon_traj = pred_trajectory[:, [1, 2, 0]]
            recon_traj[:, 0] *= -1
            recon_traj = np.vstack(([0, 0, 0], recon_traj))  # Add origin point

            # Process image-based trajectory
            img_traj = img_trajectory[:, [1, 2, 0]]
            img_traj[:, 0] *= -1
            img_traj = np.vstack(([0, 0, 0], img_traj))  # Add origin point

            # Create three separate visualizations
            img_orig, img_bev_orig = visualize_frame_img(
                img=frame.copy(),
                trajectory=original_traj,
                color=(0, 255, 0),  # Green for original
            )

            img_recon, img_bev_recon = visualize_frame_img(
                img=frame.copy(),
                trajectory=recon_traj,
                color=(0, 0, 255),  # Red for direct prediction
            )

            img_d3nav, img_bev_d3nav = visualize_frame_img(
                img=frame.copy(),
                trajectory=img_traj,
                color=(255, 0, 0),  # Blue for image-based prediction
            )

            # Stack all three vertically
            combined_cam = np.vstack([img_orig, img_recon, img_d3nav])
            combined_bev = np.vstack(
                [img_bev_orig, img_bev_recon, img_bev_d3nav]
            )

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                combined_cam,
                "Ground Truth",
                (10, 30),
                font,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                combined_cam,
                "Trajectory Quantizer Prediction",
                (10, combined_cam.shape[0] // 3 + 30),
                font,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                combined_cam,
                "D3Nav Planner",
                (10, 2 * combined_cam.shape[0] // 3 + 30),
                font,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                combined_bev,
                "Ground Truth",
                (10, 30),
                font,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                combined_bev,
                "Trajectory Quantizer Prediction",
                (10, combined_bev.shape[0] // 3 + 30),
                font,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                combined_bev,
                "D3Nav Planner",
                (10, 2 * combined_bev.shape[0] // 3 + 30),
                font,
                1,
                (255, 255, 255),
                2,
            )

            # Combine camera view and BEV horizontally
            final_vis = np.hstack([combined_cam, combined_bev])

            # Save the visualization
            # Initialize video writer using dimensions from first frame
            if video_writer is None:
                height, width = final_vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_path = f"{out_dir}/chunk_{chunk:03d}.mp4"
                video_writer = cv2.VideoWriter(
                    video_path, fourcc, fps * 2, (width, height)
                )
                print(
                    f"Created new video file with dimensions {width}x{height}"
                )

            # Write frame to video
            video_writer.write(final_vis)

        # Release video writer for this chunk and prepare for next chunk
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            print(f"Saved video chunk to {video_path}")


if __name__ == "__main__":
    main()
