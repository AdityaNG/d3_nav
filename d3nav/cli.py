"""CLI interface for d3nav project.

Launches the D3Nav
"""

import os

import cv2
import numpy as np
import torch

from .model.d3nav import (
    DEFAULT_DATATYPE,
    D3Nav,
    transform_img,
    transpose_and_clip,
)


def main(args):  # pragma: no cover
    video_path = os.path.expanduser(args.video_path)
    cap = cv2.VideoCapture(video_path)

    FRAME_QUEUE = []

    with torch.no_grad():
        model = D3Nav()

        model.print_params()

        model = model.to(device=args.device, dtype=DEFAULT_DATATYPE)

        for _ in range(args.start_frame):
            ret, frame = cap.read()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (512, 256))
            frame_np = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            frame_np = transform_img(frame_np)
            frame_np = np.transpose(frame_np, (2, 0, 1))

            FRAME_QUEUE.append(frame_np)

            if len(FRAME_QUEUE) >= args.max_frames:
                x = torch.tensor(FRAME_QUEUE[-args.max_frames :])
                x = x.to(device=args.device, dtype=DEFAULT_DATATYPE).unsqueeze(
                    0
                )
                xp, ego_trajectory = model(x)
                xq = model.quantize(x)

                img = transpose_and_clip(xp[0].cpu())[0]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                imgq = transpose_and_clip(xq[0].cpu())[0]
                imgq = cv2.cvtColor(imgq, cv2.COLOR_RGB2BGR)

                cv2.imshow("Input Image", frame)
                cv2.imshow("Quantized Input", imgq)
                cv2.imshow("Future Image", img)

                key = cv2.waitKey(args.delay)

                if ord("q") == key:
                    exit()

                FRAME_QUEUE = FRAME_QUEUE[-args.max_frames :]
