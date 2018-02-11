#!/bin/bash
#SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH -L SCRATCH
#SBATCH -N 2 
#SBATCH -t 0:30:00
#SBATCH -J cifar_horovod 
#SBATCH --output=horovod_cifar.%j.log

# load modules
module load tensorflow/intel-horovod-mpi-head
# config in hep
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=66



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


srun --no-kill --nodelist ${SLURM_WORKER_HOSTS} -n 1 -N 1 python ../resnet_imagenet_main.py \
                               --train_data_path=$SCRATCH/data \
                               --log_root=$LOG_DIR/resnet_model \
                               --train_dir=$LOG_DIR/resnet_model/train \
                               --dataset='imagenet' \
                               --batch_size=128 \
                               --train_steps=80000 \
                               --num_gpus=1 > $LOG_DIR/train.${SLURM_JOBID}.log 2>&1 &

srun --no-kill --nodelist ${SLURM_EVAL_HOSTS} -n 1 -N 1 python ../resnet_imagenet_eval.py \
                               --eval_data_path=$SCRATCH/data/imagenet \
                               --log_root=$LOG_DIR/resnet_model \
                               --eval_dir=$LOG_DIR/resnet_model/test \
                               --mode=eval \
                               --dataset='imagenet' \
                               --num_gpus=1 > $LOG_DIR/test.${SLURM_JOBID}.log 2>&1



