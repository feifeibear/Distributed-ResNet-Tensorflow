#!/bin/bash

#SBATCH --job-name=cifar
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --output=slurm_cifar_eval_%j.log

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
#module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3
module load TensorFlow/1.3.0-CrayGNU-17.08-cuda-8.0-python3

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-daint/bin/activate

DATASET=cifar10

export TF_EVAL_FLAGS="
  --eval_data_path=${SCRATCH}/data/cifar-10-batches-bin/test_batch* \
  --log_root=./tmp/resnet_model \
  --eval_dir=./tmp/resnet_model/test \
  --dataset=${DATASET} \
  --mode=eval \
  --num_gpus=1
"

python3 resnet_cifar_eval.py \
  ${TF_EVAL_FLAGS}

# deactivate virtualenv
deactivate
