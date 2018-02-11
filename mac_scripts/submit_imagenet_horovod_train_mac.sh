. /home/fangjr/Envs/horovod/bin/activate
export TF_SCRIPT="../resnet_imagenet_main_horovod.py"
export DATASET=imagenet
export TF_FLAGS="
  --train_data_path=/data2/fjr \
  --log_root=./tmp/resnet_model \
  --train_dir=./tmp/resnet_model/train \
  --dataset=${DATASET} \
  --mode=train \
  --num_gpus=1 \
  --batch_size=64 \
  --sync_replicas=True \
  --train_steps=112600 \
  --num_epochs=90 \
  --benchmark_mode=False
"

mpirun -np 4 python3 $TF_SCRIPT $TF_FLAGS

deactivate
