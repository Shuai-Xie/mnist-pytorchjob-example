# mnist-pytorchjob-example

This repo provides some example codes to launch DDP training with PytorchJob or on Bare Metal.



Code structure

```sh
.
├── ckpts       # checkpoints dir
├── data        # dataset dir
│   └── FashionMNIST
├── Dockerfile              # build PytrochJob Pod container image
├── mnist_ddp_launch.py     # DDP training with launch.py
├── mnist_ddp_mp.py         # DDP training with mp.spawn()
├── mnist_dp.py             # DP training on single machine
├── models                  # torch models
│   ├── net.py
│   └── resnet.py
├── README.md
├── run.sh                  # DDP run scripts on bare metal
├── utils.py                # util functions
└── yamls
    ├── mnist_ddp_launch_host.yaml      # DDP hostNetwork, use launch.py in command
    ├── mnist_ddp_launch.yaml           # DDP officical, no launch.py in command
    ├── mnist_ddp_mp_file.yaml          # DDP w/o hostNetwork, mp with file://
    ├── mnist_ddp_mp_tcp_host.yaml      # DDP hostNetwork, mp with env://
    └── mnist_dp.yaml                   # DP single pod
```



Experiment settings

- Two V100 GPU machines 48/49. Each has 4 GPUs. We have 8 GPUs in total.
- DDP training resnet18 on mnist dataset with batchsize=256 and epochs=1.
- set random seed=1.



## Bare Metal

please refer to `run.sh`.



## PytorchJob

### Official example 

Refer to: https://github.com/kubeflow/pytorch-operator/blob/master/examples/mnist/README.md

- Each Pod has 1 GPU.
- PytorchJob has set all the necessary environment variables to launch the DDP training. So the command of the container can be easily set as below.

```sh
# container command
[
  "python", "mnist_ddp_launch.py", "--epochs=1", "--batch-size=256",
]

# start DDP training
kubectl apply -f yamls/mnist_ddp_launch.yaml
```



### Personal Usage

In order to use PytorchJob more flexibly like on Bare Metal, We try to use the os.environ `RANK` set by PytorchJob as the `Node_RANK`. 

**Originally, this `RANK` value ranks the Pods launched by PytorchJob. Since each Pod is an independent environment, we try to directly use this `RANK` as the `Node_RANK`.**

**By this way, the container command can be set as on Bare Metal because we have `Node_RANK` set by PytorchJob now. In addition, this way can only work when `hostNetwork=true`.**



#### (1) 2 Pod * 4 GPU

1 Master + 1 Worker, Each Pod has 4 GPUs.

```sh
# container command
[
  "sh",
  "-c",
  "python -m torch.distributed.launch --nnodes=2 --nproc_per_node=4 --node_rank=${RANK} --master_addr=10.252.192.48 --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256",
]

# start DDP training
kubectl apply -f yamls/mnist_ddp_launch_host.yaml
```

Its counterpart on Bare Metal is `nnodes=2, nproc_per_node=4`

```sh
# 48
$ python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
# 49
$ python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
```



#### (2) 4 Pod * 2 GPU

1 Master + 3 Worker, Each Pod has 2 GPUs.

```sh
# container command
[
  "sh",
  "-c",
  "python -m torch.distributed.launch --nnodes=4 --nproc_per_node=2 --node_rank=${RANK} --master_addr=10.252.192.48 --master_port=33333 mnist_ddp_launch.py --epochs=1 --batch-size=256",
]

# start DDP training
$ kubectl apply -f yamls/mnist_ddp_launch_host.yaml
```

Its counterpart on Bare Metal is `nnodes=4, nproc_per_node=2`

```sh
# 48
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=0 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=1 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
# 49
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=2 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=3 --master_addr="10.252.192.48" --master_port=22222 mnist_ddp_launch.py --epochs=1 --batch-size=256
```



#### (3) 8 Pod * 1 GPU

1 Master + 7 Worker, Each Pod has 1 GPU.

```sh
# container command
[
  "sh",
  "-c",
  "python -m torch.distributed.launch --nnodes=8 --nproc_per_node=1 --node_rank=${RANK} --master_addr=10.252.192.48 --master_port=33333 mnist_ddp_launch.py --epochs=1 --batch-size=256",
]

# start DDP training
$ kubectl apply -f yamls/mnist_ddp_launch_host.yaml
```

Its counterpart on Bare Metal is `nnodes=8, nproc_per_node=1`

```sh
# 48
CUDA_VISIBLE_DEVICES=0 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=0 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=1 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=1 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=2 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=3 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=3 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
# 49
CUDA_VISIBLE_DEVICES=0 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=4 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=1 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=5 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=2 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=6 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
CUDA_VISIBLE_DEVICES=3 python mnist_ddp_mp.py --nproc_per_node=1 --nnodes=8 --node_rank=7 --dist-url="tcp://10.252.192.48:22222" --epochs=1 --batch-size=256
```

