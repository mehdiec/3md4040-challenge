from __future__ import print_function, division
import os
import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from matplotlib import cm


# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode


class PlanktonsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.name = self.data.iloc[:, 0]
        self.targets = self.data.iloc[:, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        landmarks = self.data.iloc[idx, 2:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype("float").reshape(-1, 28)
        img = Image.fromarray(np.uint8(landmarks), "L")
        target = int(self.targets[idx])

        if self.transform:
            img = self.transform(img)

        return img, target
