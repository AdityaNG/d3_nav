import cv2
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper

# from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from d3nav.model.d3nav import transform_img


class NuScenesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        nusc: NuScenes,
        is_train: bool = True,
        prediction_horizon: int = 6,
        fps: float = 2,
        val_split: float = 0.2,
        temporal_context: int = 8,
    ):
        self.nusc = nusc
        self.temporal_context = temporal_context
        self.is_train = is_train
        self.prediction_horizon = prediction_horizon
        self.fps = fps
        self.helper = PredictHelper(self.nusc)

        # Create our own train/val split
        all_scenes = self.nusc.scene
        splits = create_splits_scenes()
        scene_names = splits["train"] if is_train else splits["val"]

        self.scenes = []
        for scene in all_scenes:
            if scene["name"] in scene_names:
                self.scenes.append(scene)

        print(
            f"{'Train' if is_train else 'Validation'} "
            f"set: {len(self.scenes)} scenes"
        )
        self.samples = self._get_samples()

    def _get_samples(self):
        samples = []

        for scene in self.scenes:
            # print(f"Processing scene: {scene['name']}")
            sample_token = scene["first_sample_token"]

            while sample_token:
                sample = self.nusc.get("sample", sample_token)

                # Get image paths for the last self.temporal_context frames
                image_paths = []
                current_sample = sample
                for _ in range(self.temporal_context):
                    cam_front_data = self.nusc.get(
                        "sample_data", current_sample["data"]["CAM_FRONT"]
                    )
                    image_paths.insert(
                        0,
                        self.nusc.get_sample_data_path(
                            cam_front_data["token"]
                        ),
                    )
                    if current_sample["prev"] == "":
                        break
                    current_sample = self.nusc.get(
                        "sample", current_sample["prev"]
                    )

                if len(image_paths) == self.temporal_context:
                    # Get future trajectory
                    try:
                        future_xy = self._get_ego_future_trajectory(
                            sample_token
                        )

                        if len(future_xy) == self.prediction_horizon:
                            sample_dict = {
                                "image_paths": image_paths,
                                "trajectory": future_xy,
                                "sample_token": sample_token,
                            }

                            if not self.is_train:
                                # Load 3D bboxes for collision calculation
                                bboxes = self._get_3d_bboxes(sample_token)
                                sample_dict["3d_bboxes"] = bboxes

                            samples.append(sample_dict)
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        exit()
                        print(
                            f"Error processing sample {sample_token}: {str(e)}"
                        )

                sample_token = sample["next"]

        set_name = "training" if self.is_train else "validation"
        print(f"Loaded {len(samples)} samples for {set_name}")
        return samples

    def _get_ego_future_trajectory(self, sample_token):
        future_xy = []
        current_sample = self.nusc.get("sample", sample_token)

        # Get the initial ego pose
        initial_pose = self.nusc.get(
            "ego_pose", current_sample["data"]["LIDAR_TOP"]
        )
        initial_position = np.array(initial_pose["translation"][:3])
        initial_rotation = Quaternion(initial_pose["rotation"])

        for _ in range(self.prediction_horizon):
            if current_sample["next"] == "":
                break
            current_sample = self.nusc.get("sample", current_sample["next"])
            ego_pose = self.nusc.get(
                "ego_pose", current_sample["data"]["LIDAR_TOP"]
            )

            # Calculate relative position
            position = np.array(ego_pose["translation"][:3])
            relative_position = position - initial_position

            # Rotate the relative position to the initial frame
            rotated_position = initial_rotation.inverse.rotate(
                relative_position
            )

            future_xy.append(rotated_position)

        return np.array(future_xy)

    def _get_3d_bboxes(self, sample_token):
        sample = self.nusc.get("sample", sample_token)
        bboxes = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            # Create the box from the annotation data
            box = dict(
                center=ann["translation"],
                size=ann["size"],
                orientation=ann["rotation"],
                name=ann[
                    "category_name"
                ],  # Use 'name' instead of 'label' for the category
                token=ann["token"],
            )
            # box = Box(
            #     center=ann['translation'],
            #     size=ann['size'],
            #     orientation=Quaternion(ann['rotation']),
            #     name=ann['category_name'],  # Use 'name' instead of 'label' for the category  # noqa
            #     token=ann['token']
            # )

            # # Set velocity if available
            # if 'velocity' in ann and ann['velocity'] is not None:
            #     box.velocity = ann['velocity']

            bboxes.append(box)
        return bboxes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []
        for image_path in sample["image_paths"]:
            frame = cv2.imread(image_path)
            frame = cv2.resize(frame, (512, 256))
            frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_np = transform_img(frame_np)
            frame_np = np.transpose(frame_np, (2, 0, 1))
            frame_np = frame_np.astype(np.float32)
            frames.append(frame_np)

        x = np.stack(frames, axis=0)
        y = sample["trajectory"].astype(np.float32)  # Shape: (6, 3)

        if not self.is_train:
            return x, y, sample["3d_bboxes"]
        return x, y
