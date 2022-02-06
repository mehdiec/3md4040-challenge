from random import choices
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('work', choices=["train","test"],type=str,help="Type of work")
parser.add_argument('path', type=str)
parser.add_argument('path_test', type=str, nargs="?")
args = parser.parse_args()
#"/mounts/Datasets1/ChallengeDeep/"
if args.work == "train" : 
    os.system("python3 train.py --use_gpu --num_workers 8 --normalize --data_augment --model resnet --dataset_dir" + " " + args.path)
    #print("python3 train.py --use_gpu --num_workers 8 --normalize --data_augment --model resnet --dataset_dir" + " " + args.path)


else:
    os.system("python3 test.py --use_gpu --num_workers 8 --normalize --model resnet --paramfile "+ args.path +" --dataset_dir "+ args.path_test)
    #print("python3 test.py --use_gpu --num_workers 8 --normalize --model resnet --paramfile "+ args.path +" --dataset_dir "+ args.path_test)