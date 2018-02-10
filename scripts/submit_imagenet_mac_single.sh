export TF_SCRIPT="../resnet_imagenet_main.py"
export DATASET=imagenet
export TF_FLAGS="
  --train_data_path=.. \
  --log_root=./tmp/resnet_model \
  --train_dir=./tmp/resnet_model/train \
  --dataset=${DATASET} \
  --mode=train \
  --num_gpus=1 \
  --batch_size=128 \
  --sync_replicas=True \
  --train_steps=112600 \
  --num_epochs=90 \
  --benchmark_mode=True
"

python3 $TF_SCRIPT $TF_FLAGS
