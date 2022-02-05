import argparse
from argparse import RawTextHelpFormatter
import sys
import torch
import torchvision.transforms as transforms
from data_pre.preprocesse import load_coakroaches

from utils import test_csv, torch_summarize

parser = argparse.ArgumentParser(
    description="""
Loads and tests a model.\n
Example usage :\n
python3 test.py --model linear --paramfile ./logs/linear_4/best_model.pt
""",
    formatter_class=RawTextHelpFormatter,
)

parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU")


parser.add_argument(
    "--num_workers", type=int, default=1, help="The number of CPU threads used"
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Where to store the downloaded dataset",
    default="/mounts/Datasets1/ChallengeDeep/",
)

parser.add_argument(
    "--model",
    choices=["vanilla", "fancyCNN", "PenCNN", "densenet", "resnet"],
    action="store",
    required=True,
)


parser.add_argument(
    "--paramfile",
    type=str,
    help="Which parameters to load, contains state_dict. Should be the full model, not only the state_dict",
    required=True,
)

parser.add_argument(
    "--dir",
    type=str,
    default = "./Results/",
    help="Which directory will the result be dumped",
    required=True,
)
parser.add_argument(
    "--normalize",
    help="Which normalization to apply to the input data",
    action="store_true",
)

args = parser.parse_args()


img_size = (1, 224, 224)
num_classes = 86
batch_size = 256

use_gpu = args.use_gpu
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

##################################################### Data loading
train_augment_transforms = None


(train_loader, valid_loader, test_loader, normalization_function,) = load_coakroaches(
    0.2,
    batch_size,
    args.num_workers,
    args.normalize,
    dataset_dir=args.dataset_dir,
    train_augment_transforms=train_augment_transforms,
)


##################################################### Model building
best_model_path = args.paramfile

# model = models.build_model(args.model, img_size, num_classes)
# model.load_state_dict(torch.load(best_model_path))

model = torch.load(best_model_path)
if not issubclass(type(model), torch.nn.Module):
    print(
        """Your param file {} does not seem to be the right one.
             You should provide a model that has been fully saved with torch.save(model, filepath)
             and not only the state_dict of the model.""".format(
            best_model_path
        )
    )
    sys.exit(-1)


model = model.to(device=device)

##################################################### Evaluation

model.eval()

criterion = torch.nn.CrossEntropyLoss()

test_csv(model, test_loader, device, dir=args.dir)
