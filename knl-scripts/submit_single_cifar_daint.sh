#!/bin/bash

#SBATCH --job-name=cifar-single
#SBATCH --time=02:15:00
#SBATCH --nodes=2
#SBATCH --constraint=gpu
#SBATCH --output=s-cifar_%j.log

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
# module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3
module load TensorFlow/1.3.0-CrayGNU-17.08-cuda-8.0-python3

SLURM_WORKER_HOSTS=$(scontrol show hostnames ${SLURM_NODELIST} | 
                       head -n 1 | tr -s '\n' ',' | 
                       head --bytes -1)

SLURM_EVAL_HOSTS=$(scontrol show hostnames ${SLURM_NODELIST} | 
                       tail -n 1 | tr -s '\n' ',' | 
                       head --bytes -1)

echo "worker is $SLURM_WORKER_HOSTS"
echo "evaler is $SLURM_EVAL_HOSTS"

LOG_DIR=./single_tmp
mkdir -p $LOG_DIR 
rm -rf $LOG_DIR/*.log


srun --no-kill --nodelist ${SLURM_WORKER_HOSTS} -n 1 -N 1 python3 ../resnet_cifar_main.py \
                               --train_data_path=$SCRATCH/data \
                               --log_root=$LOG_DIR/resnet_model \
                               --train_dir=$LOG_DIR/resnet_model/train \
                               --dataset='cifar10' \
                               --batch_size=128 \
                               --train_steps=80000 \
                               --num_gpus=1 > $LOG_DIR/train.${SLURM_JOBID}.log 2>&1 &



srun --no-kill --nodelist ${SLURM_EVAL_HOSTS} -n 1 -N 1 python3 ../resnet_cifar_eval.py \
                               --eval_data_path=$SCRATCH/data/cifar-10-batches-bin/test_batch* \
                               --log_root=$LOG_DIR/resnet_model \
                               --eval_dir=$LOG_DIR/resnet_model/test \
                               --mode=eval \
                               --dataset='cifar10' \
                               --num_gpus=1 > $LOG_DIR/test.${SLURM_JOBID}.log 2>&1



