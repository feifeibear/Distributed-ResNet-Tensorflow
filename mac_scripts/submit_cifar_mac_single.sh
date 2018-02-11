export TF_SCRIPT="../resnet_cifar_main.py"
export DATASET=cifar10
export TF_FLAGS="
  --train_data_path=${HOME}/dataset \
  --log_root=./tmp/resnet_model \
  --train_dir=./tmp/resnet_model/train \
  --dataset=${DATASET} \
  --num_gpus=1 \
  --batch_size=128 \
  --sync_replicas=False \
  --train_steps=80000
"

python3 $TF_SCRIPT $TF_FLAGS
