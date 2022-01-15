import numpy as np
import cv2
import os.path
import copy
import glob
import time

# from plankton import PlanktonsDataset
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles=[-30, -15, 0, 15, 30]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])


DIM = 64, 64  # 256, 256


class TrainValidDataset(Dataset):
    def __init__(self, data_root, transform=False):
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        for image_paths in os.listdir(data_root):

            real_path = os.path.join(data_root, image_paths)
            name = int(image_paths.split("/")[-1][:3])
            boom = 1
            if len(os.listdir(real_path)) < 1999:
                boom = int(2000 / len(os.listdir(real_path)))
            for _ in range(boom):
                for filename in os.listdir(real_path):

                    img_file = os.path.join(real_path, filename)
                    self.samples.append((name, img_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        classe = self.samples[idx][0]
        image = self.samples[idx][1]
        image = Image.open(image)

        if self.transform:
            img_file = image.resize(DIM)
            img_grey = img_file.convert("L")
            img_grey = self.transform(image=img_file)["image"]
        else:
            img_file = image.resize(DIM)
            img_grey = img_file.convert("L")

        return img_grey, classe


class TestDataset(Dataset):
    def __init__(self, data_root, transform=False):

        self.transform = transform
        self.samples = []
        for filename in os.listdir(data_root):
            name = filename.split("/")[-1]
            img_file = os.path.join(data_root, filename)

            self.samples.append((name, img_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        classe = self.samples[idx][0]
        image = self.samples[idx][1]
        image = Image.open(image)
        if self.transform:
            img_file = image.resize(DIM)
            img_grey = img_file.convert("L")
            img_grey = self.transform(image=img_file)["image"]
        else:
            img_file = image.resize(DIM)
            img_grey = img_file.convert("L")
        return img_grey, classe


def compute_mean_std(loader):
    # Compute the mean over minibatches
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    # Compute the std over minibatches
    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img) ** 2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)

    # Set the variance of pixels with no variance to 1
    # Because there is no variance
    # these pixels will anyway have no impact on the final decision
    std_img[std_img == 0] = 1

    return mean_img, std_img


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


class CenterReduce:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


# save this stuff somewhere
# split in validation and training test set


def load_coakroaches(
    valid_ratio,
    batch_size,
    num_workers,
    normalize,
    train_augment_transforms=None,
):

    start_time = time.time()

    # Load the dataset for the training/validation sets

    # Split it into training and validation sets /mounts/Datasets1/ChallengeDeep/train/ /mounts/Datasets1/ChallengeDeep/test/imgs/
    train_valid_dataset = TrainValidDataset(
        data_root="/mounts/Datasets1/ChallengeDeep/train/"
    )
    nb_train, nb_valid = int((1.0 - valid_ratio) * len(train_valid_dataset)), int(
        valid_ratio * len(train_valid_dataset)
    )
    if nb_train + nb_valid != len(train_valid_dataset):
        nb_train = nb_train + 1

    print("Split is Done")
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(
        train_valid_dataset, [nb_train, nb_valid]
    )

    print("Train Val loaded")

    test_dataset = TestDataset(
        "/mounts/Datasets1/ChallengeDeep/test/imgs/",
    )
    print("Test  loaded")
    # Load the test set
    # test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir, train=False)

    # Do we want to normalize the dataset given the statistics of the training set ?
    data_transforms = {
        "train": transforms.ToTensor(),
        "valid": transforms.ToTensor(),
        "test": transforms.ToTensor(),
    }

    if train_augment_transforms:
        data_transforms["train"] = transforms.Compose(
            [train_augment_transforms, transforms.ToTensor()]
        )

    if normalize:

        normalizing_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
        normalizing_loader = torch.utils.data.DataLoader(
            dataset=normalizing_dataset, batch_size=batch_size, num_workers=num_workers
        )

        # Compute mean and variance from the training set
        mean_train_tensor, std_train_tensor = compute_mean_std(normalizing_loader)

        normalization_function = CenterReduce(mean_train_tensor, std_train_tensor)

        # Apply the transformation to our dataset
        for k, old_transforms in data_transforms.items():
            data_transforms[k] = transforms.Compose(
                [old_transforms, transforms.Lambda(lambda x: normalization_function(x))]
            )
    else:
        normalization_function = None

    train_dataset = DatasetTransformer(train_dataset, data_transforms["train"])
    valid_dataset = DatasetTransformer(valid_dataset, data_transforms["valid"])
    test_dataset = DatasetTransformer(test_dataset, data_transforms["test"])

    # shuffle = True : reshuffles the data at every epoch
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print("--- %s seconds ---" % (time.time() - start_time))

    return train_loader, valid_loader, test_loader, copy.copy(normalization_function)
