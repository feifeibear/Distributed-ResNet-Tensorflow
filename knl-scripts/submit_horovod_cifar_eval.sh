. /home/fangjr/Envs/horovod/bin/activate
export TF_SCRIPT="../resnet_cifar_eval.py"
export DATASET=cifar10
export TF_EVAL_FLAGS="
  --eval_data_path=$HOME/dataset/cifar-10-batches-bin/test_batch* \
  --log_root=./tmp/resnet_model \
  --eval_dir=./tmp/resnet_model/test \
  --dataset=${DATASET} \
  --mode=eval \
  --num_gpus=1 \
  --eval_once=True
"
python3 $TF_SCRIPT $TF_EVAL_FLAGS
deactivate



