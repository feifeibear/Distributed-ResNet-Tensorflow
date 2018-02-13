#!/bin/bash
#SBATCH -p debug 
#SBATCH -C knl,quad,cache
#SBATCH -L SCRATCH
#SBATCH -t 0:30:00
#SBATCH -J cifar_horovod 
#SBATCH --output=imagenet.%j.log

module load tensorflow/intel-horovod-mpi-head
# config in hep
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=66
export TF_ENABLE_WINOGRAD_NONFUSED=True

export PYTHON=python
let BATCH_SIZE=64
export BATCH_SIZE
export TF_SCRIPT="../resnet_imagenet_main_horovod.py"
export DATASET=imagenet
export FAKE_DATA=True
export LOG_DIR=./logs/horovod_${DATASET}_${SLURM_JOB_NUM_NODES}_nodes_${BATCH_SIZE}_batch_${FAKE_DATA}
mkdir -p $LOG_DIR

export TF_FLAGS="
  --use_horovod=True \
  --train_data_path=$SCRATCH/data/imagenet \
  --log_root=${LOG_DIR}/resnet_model \
  --train_dir=${LOG_DIR}/resnet_model/train \
  --dataset=${DATASET} \
  --num_gpus=0 \
  --batch_size=${BATCH_SIZE} \
  --sync_replicas=True \
  --train_steps=112600\
  --num_intra_threads=66 \
  --num_inter_threads=3 \
  --num_epochs=90 \
  --data_format=channels_last \
  --benchmark_mode=${FAKE_DATA}
"

echo $SLURM_JOB_NUM_NODES
mkdir -p ${LOG_DIR}
srun -n ${SLURM_JOB_NUM_NODES} -N ${SLURM_JOB_NUM_NODES} -c 272 $PYTHON $TF_SCRIPT $TF_FLAGS &> ${LOG_DIR}/performance.log
