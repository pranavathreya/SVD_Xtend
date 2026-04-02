import os
import random

import numpy as np


def get_split_folders(base_folder, split):
    validation_split=0.1
    random_seed=42
    folders = os.listdir(base_folder)
    rng = random.Random(random_seed)
    rng.shuffle(folders)
    split_idx = int(np.floor(validation_split * len(folders)))

    if split == "train":
        return folders[split_idx:]
    if split == "val":
        return folders[:split_idx]

    raise ValueError(f"Unknown split: {split}")
