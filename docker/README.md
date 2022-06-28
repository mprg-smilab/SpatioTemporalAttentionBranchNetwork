# Docker for ST-ABN


## Pull Docker Images

First, you need to pull docker images.
You can use the following two images:

* NGC container: `nvcr.io/nvidia/pytorch:21.07-py3`
* Our built image: `cumprg/stabn:1.10.0`

To pull docker images, please run the following commands.
```bash
# 1. NGC container
docker pull nvcr.io/nvidia/pytorch:21.07-py3

# 2. Our built image
docker pull cumprg/stabn:1.10.0
```

`cumprg/stabn:1.10.0` runs docker daemon as a general user with the same user ID in host OS.
The detailed usage is described below (see Run).

## Build image

If you want to build docker image `cumprg/stabn:1.10.0` on your environment by yourself, please run following command.

```bash
bash build.sh
```

## Run docker

### 1. NGC Container

Please execute `docker run` command. For example:

```bash
docker run --gpus all -ti --rm --ipc=host \
    --name=[container name] -v [volume mount] \
    nvcr.io/nvidia/pytorch:21.07-py3
```

### 2. Our built image

By using `cumprg/stabn:1.10.0`, you can run docker with a general user with the same user ID in host OS.

You can run a docker daemon by following command. (User ID is automatically set.)

```bash
./run.sh [container name] [volume mount 1] [volume mount 2] ...
```

If you want run manually, please add user ID option. For example,

```bash
docker run --gpus all -ti --rm --ipc=host \
    --name=[container name] -v [volume mount] -u [uid]:[gid] \
    cumprg/stabn:1.10.0
```

## Memo

Here, we write a memorandum to remind the execution environment for the original [ST-ABN program v0.1](https://github.com/machine-perception-robotics-group/ST-ABN_PyTorch/tree/v0.1).

### Library & Python Package Requirements

The requirements of [the original ST-ABN program](https://github.com/machine-perception-robotics-group/ST-ABN_PyTorch/tree/v0.1) is as follows:

- PyTorch: 1.7.0
- torchvision: 0.7.0
- Python: 3.5+
- NVCC: 2+
- GCC: 4.9+
- mmcv: 0.2.10

If you want to use [the original ST-ABN program](https://github.com/machine-perception-robotics-group/ST-ABN_PyTorch/tree/v0.1), you should use `nvcr.io/nvidia/pytorch:20.09-py3`.

The details of NGC container can be found at [Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/).
