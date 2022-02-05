import math
import torch
import torch.nn as nn
import torch.nn.init
import torchvision
from collections import OrderedDict

# BIEN CHOISIR LES INPUT LAYER POUR LA DERNIERE COUCHE
def linear(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out)]


def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True)]


def dropout_linear(dim_in, dim_out, p_drop):
    return [nn.Dropout(p_drop), nn.Linear(dim_in, dim_out)]


def dropout_linear_relu(dim_in, dim_out, p_drop):
    return [nn.Dropout(p_drop), nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True)]


def bn_dropout_linear(dim_in, dim_out, p_drop):
    return [nn.BatchNorm1d(dim_in), nn.Dropout(p_drop), nn.Linear(dim_in, dim_out)]


def bn_dropout_linear_relu(dim_in, dim_out, p_drop):
    return bn_dropout_linear(dim_in, dim_out, p_drop) + [nn.ReLU(inplace=True)]


def conv_relu_maxp(in_channels, out_channels, ks):
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=1,
            padding=int((ks - 1) / 2),
            bias=True,
        ),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    ]


def conv_bn_relu(in_channels, out_channels, ks):
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=1,
            padding=int((ks - 1) / 2),
            bias=True,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class VanillaCNN(nn.Module):
    def __init__(self, num_classes):
        super(VanillaCNN, self).__init__()

        # By default, Linear layers and Conv layers use Kaiming He initialization

        self.features = nn.Sequential(
            *conv_relu_maxp(1, 16, 5),
            *conv_relu_maxp(16, 32, 5),
            *conv_relu_maxp(32, 64, 5)
        )
        probe_tensor = torch.zeros((1, 1, 28, 28))
        out_features = self.features(probe_tensor).view(-1)

        self.classifier = nn.Sequential(
            *dropout_linear_relu(out_features.shape[0], 128, 0.5),
            *dropout_linear_relu(128, 256, 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)  #  OR  x = x.view(-1, self.num_features)
        y = self.classifier(x)
        return y


class FancyCNN(nn.Module):
    def __init__(self, num_classes):
        super(FancyCNN, self).__init__()

        base_n = 64
        self.features = nn.Sequential(
            *conv_bn_relu(1, base_n, 3),
            *conv_bn_relu(base_n, base_n, 3),
            nn.MaxPool2d(kernel_size=2),  # nn.Dropout(0.4),
            *conv_bn_relu(base_n, 2 * base_n, 3),
            *conv_bn_relu(2 * base_n, 2 * base_n, 3),
            nn.MaxPool2d(kernel_size=2),  # nn.Dropout(0.4),
            *conv_bn_relu(2 * base_n, 4 * base_n, 3),
            *conv_bn_relu(4 * base_n, 4 * base_n, 3),
            nn.AvgPool2d(kernel_size=7)
        )

        self.lin1 = nn.Linear(32 * 32, num_classes)

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size()[0], -1)

        y = self.lin1(nn.functional.dropout(x, 0.5, self.training, inplace=True))
        return y


class PenCNN(nn.Module):
    def __init__(self, num_classes):
        super(PenCNN, self).__init__()

        base_n = 64
        self.features = nn.Sequential(
            *conv_bn_relu(1, base_n, 3),
            *conv_bn_relu(base_n, base_n, 3),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4),
            *conv_bn_relu(base_n, 2 * base_n, 3),
            *conv_bn_relu(2 * base_n, 2 * base_n, 3),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4),
            *conv_bn_relu(2 * base_n, 4 * base_n, 3),
            *conv_bn_relu(4 * base_n, 4 * base_n, 3),
            nn.AvgPool2d(kernel_size=7)
        )

        self.lin1 = nn.Linear(32 * 32, num_classes)  # 144*144

    def penalty(self):
        return 1e-4 * self.lin1.weight.norm(2)

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size()[0], -1)
        y = self.lin1(nn.functional.dropout(x, 0.5, self.training, inplace=True))
        return y


def build_model(model_name, img_size, num_classes):
    model = None
    if model_name == "vanilla":
        model = VanillaCNN(num_classes)
    elif model_name == "fancyCNN":
        model = FancyCNN(num_classes)
    elif model_name == "PenCNN":
        model = PenCNN(num_classes)
    elif model_name == "resnet":
        resnet = torchvision.models.resnet18(pretrained=True)
        infeat = resnet.fc.in_features
        resnet.fc = nn.Linear(infeat, num_classes)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = resnet
    # does not work sad
    elif model_name == "densenet":
        densenet = torch.hub.load(
            "pytorch/vision:v0.10.0", "densenet121", pretrained=True
        )

        infeat = densenet.classifier.in_features

        densenet.classifier = nn.Sequential(nn.Linear(infeat, num_classes))
        num_init_features = 64
        densenet.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            1,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        model = densenet

    else:
        raise NotImplementedError("Unknown model {}".format(model_name))
    return model
