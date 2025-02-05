import argparse
import os
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

from d3nav.scripts.train_traj import D3NavTrainingModule
from d3nav.visual import visualize_frame_img


def process_frame(frame, model, frame_history):
    """Process a single frame through the D3Nav model."""
    # Resize frame to model input size
    frame_resized = cv2.resize(frame, (256, 128))

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Convert to tensor and normalize
    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)

    # Sample 8 frames at 2 FPS from the history
    if len(frame_history) >= 8:
        # Convert all frames in history to tensors
        history_tensors = []
        step = len(frame_history) // 8
        for i in range(0, len(frame_history), step):
            if len(history_tensors) < 8:  # Ensure we only get 8 frames
                frame = frame_history[i]
                # Resize and convert to tensor
                frame = cv2.resize(frame, (256, 128))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_t = torch.from_numpy(frame).float() / 255.0
                frame_t = frame_t.permute(2, 0, 1)
                history_tensors.append(frame_t)

        # Stack the tensors to create sequence
        sequence = torch.stack(history_tensors)
        sequence = sequence.unsqueeze(0).cuda()  # Add batch dimension
    else:
        # Fallback to repeating current frame if not enough history
        sequence = frame_tensor.unsqueeze(0).repeat(8, 1, 1, 1)
        sequence = sequence.unsqueeze(0).cuda()

    # Get trajectory prediction
    with torch.no_grad():
        trajectory = model.model(sequence)
        trajectory = trajectory[0].cpu().numpy()  # Remove batch dimension

    # Process trajectory for visualization
    traj = trajectory[:, [1, 2, 0]]
    traj[:, 0] *= -1
    traj = np.vstack(([0, 0, 0], traj))  # Add origin point

    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/d3nav/d3nav-epoch-03-val_loss-0.5128.ckpt",
        help="Path to checkpoint",
    )
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"vis/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = D3NavTrainingModule.load_from_checkpoint(args.ckpt)
    model.eval()

    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {args.video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate buffer size for 4.5 seconds of video
    buffer_size = int(4.5 * fps)
    frame_history = deque(maxlen=buffer_size)

    # Initialize video writer
    video_writer = None

    try:
        for _ in range(fps * 5):
            ret, frame = cap.read()
        # Process frames
        for _ in tqdm(range(frame_count), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # Add current frame to history
            frame_history.append(frame.copy())

            # Get trajectory prediction
            trajectory = process_frame(frame, model, frame_history)

            # Create visualization
            img_vis, img_bev = visualize_frame_img(
                img=frame.copy(),
                trajectory=trajectory,
                color=(255, 0, 0),  # Blue for prediction
            )

            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_vis, "Camera View", (10, 30), font, 1, (255, 255, 255), 2
            )
            cv2.putText(
                img_bev,
                "Bird's Eye View",
                (10, 30),
                font,
                1,
                (255, 255, 255),
                2,
            )

            # Combine camera view and BEV horizontally
            final_vis = np.hstack([img_vis, img_bev])

            # Initialize video writer if not already done
            if video_writer is None:
                height, width = final_vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_path = f"{out_dir}/output.mp4"
                video_writer = cv2.VideoWriter(
                    video_path, fourcc, fps, (width, height)
                )

            # Write frame to video
            video_writer.write(final_vis)

    finally:
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Output video saved to {video_path}")


if __name__ == "__main__":
    main()
