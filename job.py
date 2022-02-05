#!/usr/bin/python

import os


def makejob(model, nruns):
    return f"""#!/bin/bash 

#SBATCH --job-name=hihi-{model}
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=/usr/users/gpusdi1/gpusdi1_25/logslurms/slurm-%A_%a.out
#SBATCH --error=/usr/users/gpusdi1/gpusdi1_25/logslurms/slurm-%A_%a.err
#SBATCH --array=0-{nruns}

cd /usr/users/gpusdi1/gpusdi1_25/3md4040-challenge 
python3 src/train.py --use_gpu --model resnet --normalize --num_workers 8 --data_augment
"""


def submit_job(job):
    with open("/usr/users/gpusdi1/gpusdi1_25/job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch /usr/users/gpusdi1/gpusdi1_25/job.sbatch")


# Ensure the log directory exists
os.system("mkdir -p /usr/users/gpusdi1/gpusdi1_25/logslurms")

# Launch the batch jobs
submit_job(makejob("resnet", 0))
