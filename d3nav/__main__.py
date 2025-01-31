"""Entry point for d3nav."""

from d3nav.cli import main  # pragma: no cover

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="D3Nav video processing")
    parser.add_argument(
        "--video_path",
        type=str,
        default="/home/aditya/Videos/ams.val.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=30 * 45,
        help="Frame number to start processing from",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=8,
        help="Maximum number of frames to process at once",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (cuda or cpu)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=10,
        help="Delay between frames in milliseconds",
    )

    args = parser.parse_args()
    main(args)
