from random import choices
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('work', choices=["train","test"],type=str,help="Type of work")
parser.add_argument('path', type=str) # This argument has a double meaning, if you choose train for the work argument, you will have to put for this argument the train set path and if you choose test it will be the path of best_model.pt file 
parser.add_argument('path_test', type=str, nargs="?") # the nargs allow to not consider the argument if it's not defined
args = parser.parse_args()
# Exemple for train & test set path for the challenge "/mounts/Datasets1/ChallengeDeep/". Our code automatically checks in train and test folder
# of the path
if args.work == "train" : 
    os.system("python3 src/train.py --use_gpu --num_workers 8 --normalize --data_augment --model resnet --dataset_dir" + " " + args.path)



else:
    os.system("python3 src/test.py --use_gpu --num_workers 8 --normalize --model resnet --paramfile "+ args.path +" --dataset_dir "+ args.path_test)

