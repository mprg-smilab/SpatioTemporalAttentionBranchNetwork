# Spatio-Temporal Attention Branch Network (ST-ABN)

This repository is PyTorch implementation of Spatio-Temporal Attention Branch Network [1].

## TODO

- [ ] Inference with multiple video clips.

## Citation

If you find this repository is useful, please cite the following reference.

```bibtex
@article{Mitsuhara2021,
    author={Masahiro Mitsuhara and Tsubasa Hirakawa and Takayoshi Yamashita and Hironobu Fujiyoshi},
    title={ST-ABN: Visual Explanation Taking into Account Spatio-temporal Information for Video Recognition},
    journal={arXiv preprint, arXiv:2110.15574},
    year={2021}
}
```

## Environment

Please check README.md in docker directory.

## Data Preparation

TBA

## Training

We prepare the following two kinds of training script.

1. PyTorch Data Parallel (DP): `train_dp.py`
2. PyTorch Distributed Data Parallel (DDP): `train_ddp.py`

We recommend to use `train_ddp.py` because it is faster than `train_dp.py`.

The execution commands for each script is provided as follows.

### 1. use Data Parallel (DP)

Please see `./script_local/run_train_dp.sh`.

### 2. use Distributed Data Parallel (DDP)

Please see `./script_local/run_train_ddp.sh`

## Evaluation

Please see `./script_local/run_eval.sh`.

## Pretrained model

We publish pre-trained models.
Please download from the following link.

* [ST-ABN (ResNet 50 backbone)](https://drive.google.com/file/d/1YhXGD2NPQJVPgr5Pzm13UVr8IM9RoVwb/view?usp=sharing)
* [3D ResNet 50](https://drive.google.com/file/d/1QZURM3cr_SxkDr0axczuV0kQhRJxLPBn/view?usp=sharing)
* [3D ResNet 101](https://drive.google.com/file/d/1hE8u-DYIaQafIgfK2MmiIF_zTBUJ4TXM/view?usp=sharing)

**NOTE**

The Top-1 and Top-5 accuracies are different with ones described in the original paper because this repository contains fully re-implementaed programs and provides new trained model files.

## References

[1]. M. Mitsuhara, T. Hirakawa, T. Yamashita, H. Fujiyoshi, "ST-ABN: Visual Explanation Taking into Account Spatio-temporal Information for Video Recognition," arXiv preprint, arXiv:2110.15574, 2021.
