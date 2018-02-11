# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
#module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3
module load TensorFlow/1.3.0-CrayGNU-17.08-cuda-8.0-python3

DATASET=imagenet
export TF_EVAL_FLAGS="
  --eval_data_path=${SCRATCH}/data/imagenet \
  --log_root=./tmp/resnet_model \
  --train_dir=./tmp/resnet_model/test \
  --dataset=${DATASET} \
  --mode=eval \
  --num_gpus=1
"

python3 resnet_imagenet_eval.py \
  ${TF_EVAL_FLAGS}
