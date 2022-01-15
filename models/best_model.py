import argparse
import os
import sys

import torch
import torch.nn as nn


import utils
from data_pre.preprocesse import load_coakroaches


parser = argparse.ArgumentParser()

parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU")

parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Where to store the downloaded dataset",
    default=None,
)

parser.add_argument(
    "--num_workers", type=int, default=1, help="The number of CPU threads used"
)

parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")

parser.add_argument(
    "--data_augment",
    help="Specify if you want to use data augmentation",
    action="store_true",
)

parser.add_argument(
    "--normalize",
    help="Which normalization to apply to the input data",
    action="store_true",
)

parser.add_argument(
    "--logdir",
    type=str,
    default="./logs",
    help="The directory in which to store the logs",
)

parser.add_argument(
    "--model",
    choices=["vanilla", "fancyCNN", "PenCNN"],
    action="store",
    required=True,
)

args = parser.parse_args()


img_width = 64
img_height = 64
img_size = (1, img_height, img_width)
batch_size = 128

valid_ratio = 0.2

if args.use_gpu:
    print("Using GPU{}".format(torch.cuda.current_device()))
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

# Where to store the logs
logdir = "./logs/PenCNN_9"
print("Logging to {}".format(logdir))
if not os.path.exists(args.logdir):
    os.mkdir(args.logdir)
if not os.path.exists(logdir):
    os.mkdir(logdir)

# FashionMNIST dataset
train_augment_transforms = None


(train_loader, valid_loader, test_loader, normalization_function,) = load_coakroaches(
    valid_ratio,
    batch_size,
    args.num_workers,
    args.normalize,
    dataset_dir=args.dataset_dir,
    train_augment_transforms=train_augment_transforms,
)


loss = nn.CrossEntropyLoss()  # This computes softmax internally


print("Loading and testing the best model")

best_model_path = logdir + "/best_model.pt"
loaded_dict = torch.load(best_model_path)

model = loaded_dict["model"].to(device)
model.eval()

val_loss, val_acc, val_f1 = utils.test(model, valid_loader, loss, device)
torch.save(model, logdir + "/best_model.pt")
print(
    "Loss : {:.4f}, Acc : {:.4f}, macro F1 :  {:.4f}".format(val_loss, val_acc, val_f1)
)

sample = {"name": logdir, "Loss": val_loss, "Acc": val_acc, "macro F1": val_f1}
import json

with open("result.json", "w") as fp:
    json.dump(sample, fp)
