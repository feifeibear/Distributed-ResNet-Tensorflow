#!/bin/bash
#SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH -L SCRATCH
#SBATCH -t 0:30:00
#SBATCH -J cifar_horovod 
#SBATCH --output=horovod_cifar.%j.log

module load tensorflow/intel-horovod-mpi-head
# config in hep
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=66

PYTHON=python
let BATCH_SIZE=128/${SLURM_NNODES}
export BATCH_SIZE
export TF_SCRIPT="../resnet_cifar_main_horovod.py"
export DATASET=cifar10
export LOG_DIR=./logs/horovod_cifar_$DATASET_${SLURM_NNODES}_nodes_$BATCH_SIZE_log

export TF_FLAGS="
  --use_horovod=True \
  --train_data_path=$SCRATCH/data \
  --log_root=${LOG_DIR}/resnet_model \
  --train_dir=${LOG_DIR}/resnet_model/train \
  --dataset=${DATASET} \
  --num_gpus=0 \
  --batch_size=${BATCH_SIZE} \
  --sync_replicas=True \
  --train_steps=80000 \
  --num_intra_threads=33 \
  --num_inter_threads=2
"

srun -n ${SLURM_NNODES} -N ${SLURM_NNODES} -c 272 $PYTHON $TF_SCRIPT $TF_FLAGS

