export TF_SCRIPT="../resnet_cifar_main_horovod.py"
export DATASET=cifar10
export TF_FLAGS="
  --use_horovod=True \
  --train_data_path=$HOME/dataset \
  --log_root=./tmp/resnet_model \
  --train_dir=./tmp/resnet_model/train \
  --dataset=${DATASET} \
  --num_gpus=0 \
  --batch_size=10 \
  --sync_replicas=True \
  --train_steps=80000
"

mpirun -np 4 python3 $TF_SCRIPT $TF_FLAGS
