
import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import utils
import ann
from data_pre.preprocesse import load_coakroaches

import json


parser = argparse.ArgumentParser()

parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU")

parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Where to store the downloaded dataset",
    default="/mounts/Datasets1/ChallengeDeep/",
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
    choices=["vanilla", "fancyCNN", "PenCNN", "resnet", "densenet"],
    action="store",
    required=True,
)

args = parser.parse_args()

img_width, img_height = 224, 224

img_size = (1, img_height, img_width)
num_classes = 86
batch_size = 128
epochs = 40
valid_ratio = 0.2

if args.use_gpu:
    print("Using GPU{}".format(torch.cuda.current_device()))
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

# Where to store the logs
logdir = utils.generate_unique_logpath(args.logdir, args.model)
print("Logging to {}".format(logdir))
if not os.path.exists(args.logdir):
    os.mkdir(args.logdir)
if not os.path.exists(logdir):
    os.mkdir(logdir)

# Data augmentation
train_augment_transforms = None
if args.data_augment:
    train_augment_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        ]
    )

(train_loader, valid_loader, test_loader, normalization_function,) = load_coakroaches(
    valid_ratio,
    batch_size,
    args.num_workers,
    args.normalize,
    dataset_dir=args.dataset_dir,
    train_augment_transforms=train_augment_transforms,
)

# Init model, loss, optimizer
model = ann.build_model(args.model, img_size, num_classes)
model = model.to(device)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=0)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

# Where to save the logs of the metrics
history_file = open(logdir + "/history", "w", 1)
history_file.write(
    "Epoch\tTrain loss\tTrain acc\tVal loss\tVal acc\n"  # \tTest loss\tTest acc\n"
)

# Generate and dump the summary of the model
model_summary = utils.torch_summarize(model)


summary_file = open(logdir + "/summary.txt", "w")
summary_text = """
Executed command
===============
{}
Dataset
=======
Train transform : {}
Normalization : {}
Model summary
=============
{}
{} trainable parameters
Optimizer
========
{}
""".format(
    " ".join(sys.argv),
    train_augment_transforms,
    args.normalize,
    str(model).replace("\n", "\n\t"),
    sum(p.numel() for p in model.parameters() if p.requires_grad),
    str(optimizer).replace("\n", "\n\t"),
)
summary_file.write(summary_text)
summary_file.close()

tensorboard_writer = SummaryWriter(log_dir=logdir)
tensorboard_writer.add_text("Experiment summary", summary_text)
model_checkpoint = utils.ModelCheckpoint(
    logdir + "/best_model.pt",
    {"model": model, "normalization_function": normalization_function},
)
# Add the graph of the model to the tensorboard
inputs, _ = next(iter(train_loader))
inputs = inputs.to(device)
inputs = inputs.float()
tensorboard_writer.add_graph(model, inputs)
########################################### Main Loop ###########################################
for t in range(1):
    print("Epoch {}".format(t))
    train_loss, train_acc = utils.train(model, train_loader, loss, optimizer, device)

    val_loss, val_acc, val_f1 = utils.test(model, valid_loader, loss, device)
    print(
        "Loss : {:.4f}, Acc : {:.4f}, macro F1 :  {:.4f}".format(
            val_loss, val_acc, val_f1
        )
    )

    history_file.write(
        "{}\t{}\t{}\t{}\t{}\n".format(t, train_loss, train_acc, val_loss, val_acc)
    )
    model_checkpoint.update(val_loss)
    tensorboard_writer.add_scalar("metrics/train_loss", train_loss, t)
    tensorboard_writer.add_scalar("metrics/train_acc", train_acc, t)
    tensorboard_writer.add_scalar("metrics/val_loss", val_loss, t)
    tensorboard_writer.add_scalar("metrics/val_acc", val_acc, t)
    tensorboard_writer.add_scalar("metrics/val_f1", val_f1, t)
    sample = {"name": logdir, "Loss": val_loss, "Acc": val_acc, "macro F1": val_f1}

    with open(logdir + f"/result__{t}.json", "w") as fp:
        json.dump(sample, fp)


# Loading the best model found

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
utils.test_csv(model, test_loader, device, dir=logdir)

sample = {"name": logdir, "Loss": val_loss, "Acc": val_acc, "macro F1": val_f1}
import json

with open("result_true.json", "w") as fp:
    json.dump(sample, fp)
