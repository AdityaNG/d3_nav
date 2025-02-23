import os

import gdown
import torch

from .model.d3nav import D3Nav

gdrive_checkpoints = {
    "d3nav-epoch-15-val_loss-0.6955.ckpt": "1BIuW3Ler_AcpZPStRolIX4c52zZdTO2k",
    "d3nav-epoch-10-val_loss-0.9685.ckpt": "1PWwl3BOLJAQnknB4W06di4GVyaV9H2np",
}


def load_d3nav(ckpt_path: str = list(gdrive_checkpoints.keys())[0]) -> D3Nav:
    if os.path.isfile(ckpt_path):
        pass
    elif ckpt_path in gdrive_checkpoints:
        print("D3Nav: Using downloaded checkpoint")
        cache_dir = os.path.expanduser("~/.cache/d3nav")
        os.makedirs(cache_dir, exist_ok=True)
        gdrive_id = gdrive_checkpoints[ckpt_path]

        ckpt_path = os.path.join(cache_dir, ckpt_path)

        if not os.path.exists(ckpt_path):
            gdown.download(id=gdrive_id, output=ckpt_path, quiet=False)

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Remove 'model.' prefix from all keys
    new_state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

    model = D3Nav()
    model.load_state_dict(new_state_dict)
    model.eval()

    return model
