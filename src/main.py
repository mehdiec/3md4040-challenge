from random import choices
from typing_extensions import Required

import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('work', choices=["train","test"],type=str,help="Type of work")
parser.add_argument('path', type=str,nargs='?', default = "/mnt/d/Kaggle/Data/train2",help="Location of the training set")
parser.add_argument('path_test', type=str, nargs='?', default = "/mnt/d/Kaggle/Data/test/imgs",help="Location of the test set")
args = parser.parse_args()

if args.work == "train" : 
    os.system("python3 train.py --use_gpu --num_workers 8 --normalize --data_augment --model resnet --dataset_dir" + " " + args.path)
    #print("python3 train.py --use_gpu --num_workers 8 --normalize --data_augment --model resnet --dataset_dir" + " " + args.path)


else:
    os.system("python3 test.py --use_gpu --num_workers 8 --normalize --paramfile --model resnet --paramfile "+ args.path +" --dataset_dir "+ args.path_test)
    #print("python3 test.py --use_gpu --num_workers 8 --normalize --paramfile --model resnet --paramfile "+ args.path +" --dataset_dir "+ args.path_test)