import argparse
import os
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

from d3nav.factory import load_d3nav
from d3nav.model.d3nav import center_crop, d3nav_transform_img
from d3nav.visual import visualize_frame_img


def process_frame(frame, model, frame_history, fps):
    """Process a single frame through the D3Nav model."""

    # Sample 8 frames at 2 FPS from the history
    assert len(frame_history) >= int(4.5 * fps)

    # Convert all frames in history to tensors
    history_tensors = []
    step = len(frame_history) // 8
    for i in range(0, len(frame_history), step):
        if len(history_tensors) < 8:  # Ensure we only get 8 frames
            frame = frame_history[i]
            # Resize and convert to tensor
            frame = d3nav_transform_img(frame)
            frame_t = torch.from_numpy(frame)
            history_tensors.append(frame_t)

    # Stack the tensors to create sequence
    sequence = torch.stack(history_tensors)
    sequence = sequence.unsqueeze(0).cuda()  # Add batch dimension

    # Get trajectory prediction
    with torch.no_grad():
        trajectory = model(sequence)
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
        default="checkpoints/d3nav/d3nav-epoch-06-val_loss-0.7179.ckpt",
        help="Path to checkpoint",
    )
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"vis/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    model = load_d3nav(args.ckpt)
    model = model.cuda()
    model.temporal_context = 2
    model.eval()

    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {args.video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate buffer size for 5 seconds of video
    buffer_size = int(5 * fps)
    buffer_full = int(4.5 * fps)
    frame_history = deque(maxlen=buffer_size)

    print("buffer_size", buffer_size)

    # Initialize video writer
    video_writer = None

    crop_ratio = 0.3

    model.dropout_rate = 0.0

    try:
        for _ in range(fps * 30):
            ret, frame = cap.read()
        # Process frames
        for index in tqdm(range(frame_count), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            frame = center_crop(frame, crop_ratio)
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Add current frame to history
            frame_history.append(frame.copy())

            if len(frame_history) < buffer_full:
                continue

            # Get trajectory prediction
            trajectory = process_frame(frame, model, frame_history, fps)

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

            if index % 15 == 0:
                cv2.imwrite("vis/vis.png", final_vis)

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
