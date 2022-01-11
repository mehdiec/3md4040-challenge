import torch
import torchvision.transforms as transforms
from torchvision.transforms import RandomAffine

import numpy as np
import cv2
import os.path
import copy
from sklearn.model_selection import StratifiedShuffleSplit

# from plankton import PlanktonsDataset
from torch.utils.data import Dataset, Subset
import pandas as pd

TRAIN_CONST = [
    "000_Candaciidae",
    "043_larvae__Annelida",
    "001_detritus",
    "044_Rhopalonema",
    "002_Calocalanus_pavo",
    "045_egg__other",
    "003_larvae__Crustacea",
    "046_tail__Appendicularia",
    "004_Podon",
    "047_Euchirella",
    "005_Sapphirinidae",
    "048_calyptopsis",
    "006_Calanidae",
    "049_Haloptilus",
    "007_zoea__Decapoda",
    "050_eudoxie__Diphyidae",
    "008_Gammaridea",
    "051_egg__Actinopterygii",
    "009_Oikopleuridae",
    "052_nectophore__Diphyidae",
    "010_Hyperiidea",
    "053_head",
    "011_zoea__Galatheidae",
    "054_Penilia",
    "012_nectophore__Physonectae",
    "055_egg__Cavolinia_inflexa",
    "013_Rhincalanidae",
    "056_Pontellidae",
    "014_Acantharea",
    "057_Coscinodiscus",
    "015_Foraminifera",
    "058_Acartiidae",
    "016_nauplii__Crustacea",
    "059_Corycaeidae",
    "017_gonophore__Diphyidae",
    "060_artefact",
    "018_metanauplii",
    "061_cirrus",
    "019_megalopa",
    "062_Luciferidae",
    "020_Brachyura",
    "063_Limacinidae",
    "021_tail__Chaetognatha",
    "064_cyphonaute",
    "022_Doliolida",
    "065_part__Copepoda",
    "023_Scyphozoa",
    "066_Fritillariidae",
    "024_Ctenophora",
    "067_Echinoidea",
    "025_Bivalvia__Mollusca",
    "068_Neoceratium",
    "026_ephyra",
    "069_Phaeodaria",
    "027_Temoridae",
    "070_Ostracoda",
    "028_scale",
    "071_Centropagidae",
    "029_Evadne",
    "072_Ophiuroidea",
    "030_Copilia",
    "073_nauplii__Cirripedia",
    "031_Eucalanidae",
    "074_Salpida",
    "032_Pyrosomatida",
    "075_Oithonidae",
    "033_nectophore__Abylopsis_tetragona",
    "076_eudoxie__Abylopsis_tetragona",
    "034_Actinopterygii",
    "077_cypris",
    "035_Creseidae",
    "078_Oncaeidae",
    "036_Calanoida",
    "079_gonophore__Abylopsis_tetragona",
    "037_Decapoda",
    "080_Harpacticoida",
    "038_Obelia",
    "081_Cavoliniidae",
    "039_Noctiluca",
    "082_Aglaura",
    "040_Spumellaria",
    "083_Euchaetidae",
    "041_Chaetognatha",
    "084_Tomopteridae",
    "042_Annelida",
    "085_Limacidae",
]
DIM = 100, 100


class PlanktonsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, test=False):
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
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        landmarks = self.data.iloc[idx, 2:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype("float").reshape(-1, 28)

        if self.test:
            img = landmarks
            target = self.data.iloc[idx, 1]

        else:
            img, target = landmarks, int(self.targets[idx])

        return img, target


def compute_mean_std(loader):
    # Compute the mean over minibatches
    mean_img = None
    for imgs, _, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    # Compute the std over minibatches
    std_img = torch.zeros_like(mean_img)
    for imgs, _, _ in loader:
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


def transform_images_from_folder(train_dir):
    images = []
    for folder in train_dir:
        print(folder)

        classe = int(folder[:3])
        real_folder = "data/train/" + folder
        for filename in os.listdir(real_folder):
            img = cv2.imread(os.path.join(real_folder, filename))
            if img is not None:

                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Normalize, rescale entries to lie in [0,1]
                gray_img = gray_img.astype("float32") / 255
                resized = cv2.resize(gray_img, DIM, interpolation=cv2.INTER_AREA)
                images.append((resized, classe))
    return images


# save this stuff somewhere
# split in validation and training test set
def make_stratified_split(train_valid_dataset, valid_ratio=0.2):

    df = pd.read_csv("data/train/train_csv/train.csv")
    target = df["img_name"]
    x = df["img_class"]

    # 2.
    split = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=42)
    for train_index, test_index in split.split(x, target):
        train_index = train_index
        test_index = test_index

    print("Train size: {}\nValid size: {}".format(len(train_index), len(test_index)))

    return [
        Subset(train_valid_dataset, train_index),
        Subset(train_valid_dataset, test_index),
    ]


def load_coakroaches(
    valid_ratio,
    batch_size,
    num_workers,
    normalize,
    dataset_dir=None,
    train_augment_transforms=None,
    normalizing_tensor_path=None,
):

    if not dataset_dir:
        dataset_dir = os.path.join(
            os.path.expanduser("~"), "train", "train_csv", "train.csv"
        )

    # Load the dataset for the training/validation sets
    train_valid_dataset = PlanktonsDataset(
        csv_file="data/train/train_csv/train.csv", root_dir="data/train/"
    )

    # Split it into training and validation sets

    train_dataset, valid_dataset = make_stratified_split(
        train_valid_dataset, valid_ratio=0.2
    )
    test_dataset = PlanktonsDataset(
        csv_file="data/test/test_csv/test.csv", root_dir="data/test/", test=True
    )
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

    return train_loader, valid_loader, test_loader, copy.copy(normalization_function)


def display_tensor_samples(tensor_samples, labels=None, filename=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig = plt.figure(figsize=(20, 5), facecolor="w")
    nsamples = tensor_samples.shape[0]
    for i in range(nsamples):
        ax = plt.subplot(1, nsamples, i + 1)
        plt.imshow(tensor_samples[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
        # plt.axis('off')
        if labels is not None:
            ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def display_samples(loader, nsamples, filename=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    imgs, labels, _ = next(iter(train_loader))
    display_tensor_samples(imgs[:nsamples], labels[:nsamples], filename)
    # print("imgs is of shape {},  labels of shape {}'".format(imgs.shape, labels.shape))

    # fig=plt.figure(figsize=(20,5),facecolor='w')
    # for i in range(nsamples):
    #    ax = plt.subplot(1,nsamples, i+1)
    #    plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
    #    #plt.axis('off')
    #    ax.set_title("{}".format(classes_names[labels[i]]), fontsize=15)
    #    ax.get_xaxis().set_visible(False)
    #    ax.get_yaxis().set_visible(False)
    # if filename:
    #    plt.savefig(filename, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":

    num_threads = 4
    valid_ratio = 0.2
    batch_size = 128
    classes_names = TRAIN_CONST

    train_loader, valid_loader, test_loader, _ = load_coakroaches(
        valid_ratio, batch_size, num_threads, False
    )

    print(
        "The train set contains {} images, in {} batches".format(
            len(train_loader.dataset), len(train_loader)
        )
    )
    print(
        "The validation set contains {} images, in {} batches".format(
            len(valid_loader.dataset), len(valid_loader)
        )
    )
    # print(
    #    "The test set contains {} images, in {} batches".format(
    #        len(test_loader.dataset), len(test_loader)
    #    )
    # )

    # display_samples(train_loader, 10, 'fashionMNIST_samples.png')

    ###################################################################################
    ## Data augmentation

    train_valid_dataset = PlanktonsDataset(
        csv_file="data/train/train_csv/train.csv", root_dir="data/train/"
    )

    train_augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ]
    )

    # data augment a single sample several times
    # img, label = train_valid_dataset[np.random.randint(len(train_valid_dataset))]
    # Timg = transforms.functional.to_tensor(img)
    # n_augmented_samples = 10
    # aug_imgs = torch.zeros(
    #    n_augmented_samples, Timg.shape[0], Timg.shape[1], Timg.shape[2]
    # )
    # for i in range(n_augmented_samples):
    #    img = Image.fromarray(np.uint8(cm.gist_earth(img) * 255))
    #    aug_imgs[i] = transforms.ToTensor()(train_augment(img))
    # print("I augmented a {}".format(classes_names[label]))
    # display_tensor_samples(aug_imgs, filename="fashionMNIST_sample_augment.png")

    # Test with data augmentation
    train_augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            RandomAffine(degrees=10, translate=(0.1, 0.1)),
        ]
    )

    train_loader, _, _, _ = load_coakroaches(
        valid_ratio,
        batch_size,
        num_threads,
        False,
        train_augment_transforms=train_augment,
    )
    # display_samples(train_loader, 10, "fashionMNIST_samples_augment.png")

    # Loading normalized datasets
    train_loader, valid_loader, _, _ = load_coakroaches(
        valid_ratio, batch_size, num_threads, True
    )
