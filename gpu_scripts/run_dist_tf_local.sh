#!/bin/bash

# Arguments:
#   $1: job_name: either "ps" or "worker"
#   $2: task_index: index of `job_name` instance (starting from 0)
#   $3: TF_PS_HOSTS: Parameter Servers' hostnames
#   $4: TF_WORKER_HOSTS: Workers' hostnames

# set TensorFlow script parameters
TF_DIST_FLAGS=" --ps_hosts=$3 --worker_hosts=$4"

TF_SCRIPT=../resnet_cifar_main.py

TF_FLAGS="--train_data_path=.. \
  --log_root=./tmp/resnet_model \
  --train_dir=./tmp/resnet_model/train \
  --dataset=cifar10 \
  --num_gpus=0 \
  --batch_size=10 \
  --sync_replicas=True \
  --train_steps=100
"

# load virtualenv
# export WORKON_HOME=$HOME/Envs
# source $WORKON_HOME/tf-local/bin/activate

# train inception
python3 ${TF_SCRIPT} --job_name=$1 --task_index=$2 ${TF_DIST_FLAGS} ${TF_FLAGS}

# deactivate virtualenv
# deactivate
