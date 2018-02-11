. /home/fangjr/Envs/horovod/bin/activate
# mpirun -np 4 python3 ./cifar10_main.py --batch_size=32 --resnet_size=50
mpirun -np 4 python3 ./imagenet_main.py --batch_size=64 --resnet_size=50 --data_dir=/data2/fjr
deactivate
