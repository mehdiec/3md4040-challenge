import csv
import json
import numpy as np
import torch
from torch.nn.modules.module import _addindent
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def scores(y_pred, y_true):
    scores = []

    print(f"macro F1 : {f1_score(y_true, y_pred, average='macro')}")
    print(f"accuracy : {accuracy_score(y_true, y_pred)}")
    np_y_true = np.array(y_true)
    np_y_pred = np.array(y_pred)
    for cls in np.unique(np_y_true):
        biclass_true = (np_y_true == cls).astype(int)
        biclass_pred = (np_y_pred == cls).astype(int)
        tp = (biclass_true * biclass_pred).sum()
        fp = ((1 - biclass_true) * biclass_pred).sum()
        fn = (biclass_true * (1 - biclass_pred)).sum()
        tn = ((1 - biclass_true) * (1 - biclass_pred)).sum()
        dict_class = {
            "Class": cls,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "precision": precision_score(biclass_true, biclass_pred),
            "recall": recall_score(biclass_true, biclass_pred),
            "local_F1": f1_score(biclass_true, biclass_pred),
        }
        scores.append(dict_class)

    return scores


def train(model, loader, f_loss, optimizer, device):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    use_gpu  -- Boolean, whether to use GPU
    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()
    N = 0
    tot_loss, correct = 0.0, 0
    for i, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()
        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Accumulate the number of processed samples
        N += inputs.shape[0]

        # For the total loss
        tot_loss += inputs.shape[0] * loss.item()

        # For the total accuracy
        predicted_targets = outputs.argmax(dim=1)
        correct += (predicted_targets == targets).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        try:
            model.penalty().backward()
        except AttributeError:
            pass
        optimizer.step()

        # Display status
        progress_bar(
            i,
            len(loader),
            msg="Loss : {:.4f}, Acc : {:.4f}".format(tot_loss / N, correct / N),
        )
    return tot_loss / N, correct / N


def test(model, loader, f_loss, device):
    """
    Test a model by iterating over the loader
    Arguments :
        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- a torch.device object
    Returns :
        A tuple with the mean loss and mean accuracy
    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct, f11 = 0.0, 0, 0.0
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()

            outputs = model(inputs)

            loss = f_loss(outputs, targets)

            N += inputs.shape[0]

            # For the loss
            tot_loss += inputs.shape[0] * loss.item()

            # For the accuracy
            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()

            y_true = targets.cpu().data.numpy()
            y_pred = predicted_targets.cpu().data.numpy()

            f11 += f1_score(y_true, y_pred, average="macro")
            data = scores(y_true, y_pred)

            with open("app.json", "w") as f:
                json.dump(data, f)

        print(f"macro F1 : {f11/N}")
        print(f11, N)
        scores(y_pred, y_true)

        return tot_loss / N, correct / N


def test_csv(model, loader, device, dir):
    """
    Test a model by iterating over the loader
    Arguments :
        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- a torch.device object
    Returns :
        A tuple with the mean loss and mean accuracy
    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct, f11 = 0.0, 0, 0.0
        for _, (inputs, _) in enumerate(loader):
            inputs = inputs.to(device)
            inputs = inputs.float()

            outputs = model(inputs)
            input_cpu = inputs.cpu().data.numpy()
            names = [inp.name for inp in input_cpu]

            # For the accuracy
            predicted_targets = outputs.argmax(dim=1)

            y_pred = predicted_targets.cpu().data.numpy()
            l = [names, y_pred]

            data = zip(*l)

            with open(dir + "results.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerows(data)


# from https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py

import os
import sys
import time
import math

_, term_width = os.popen("stty size", "r").read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [{}>{}]".format("=" * cur_len, "." * rest_len))

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %10s" % format_time(step_time))
    L.append(" | Tot: %10s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint:
    def __init__(self, filepath, dict_to_save):
        self.min_loss = None
        self.filepath = filepath
        self.dict_to_save = dict_to_save

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            # torch.save(self.model.state_dict(), self.filepath)
            torch.save(self.dict_to_save, self.filepath)
            self.min_loss = loss


# https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + " (\n"
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential,
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        total_params += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += "  (" + key + "): " + modstr
        if show_weights:
            tmpstr += ", weights={}".format(weights)
        if show_parameters:
            tmpstr += ", parameters={}".format(params)
        tmpstr += "\n"

    tmpstr = tmpstr + ")"
    tmpstr += "\n {} learnable parameters".format(total_params)
    return tmpstr


# SquarePadding : https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10


class SquarePad:
    def __call__(self, image):
        max_wh = None  # Max longueur largeur des images du dataset à determiner
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, "constant")  # valeur 0 pour la couleur noir


"""
Use example :

target_image_size = (224, 224)  # as an example
# now use it as the replacement of transforms.Pad class
transform=transforms.Compose([
    SquarePad(),
    transforms.Resize(target_image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
"""
