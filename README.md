# Str-GCL: Structural Commonsense Driven Graph Contrastive Learning

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://www2025.thewebconf.org/" alt="Conference">
        <img src="https://img.shields.io/badge/TheWebConf'25-brightgreen" /></a>
<!--     <img src="https://img.shields.io/pypi/l/torch-rechub"> -->
</p>

The official source code for [**Str-GCL: Structural Commonsense Driven Graph Contrastive Learning**](https://dl.acm.org/doi/10.1145/3696410.3714900) at WWW 2025.

Part of code is referenced from [*Deep Graph Contrastive Representation Learning*](https://github.com/CRIPAC-DIG/GRACE) and [*PyGCL: A PyTorch Library for Graph Contrastive Learning*](https://github.com/PyGCL/PyGCL))

## Requirements
- python=3.9.21
- torch=2.4.0 (with CUDA 12.4 support)
- torch-geometric=2.6.1
- scikit-learn=1.6.1
- numpy=2.0.1
- scipy=1.13.1


## How to run

To get started, unzip the datasets (can be found in ./datasets), and then, enter the `scripts` directory. From there, choose the relevant script that corresponds to the dataset you're working with.
```
bash Cora.sh  # Using the Cora as an example
```

> Dear Readers,
>
> We have noticed a typographical error in the hyperparameter settings for different datasets, as presented in Table 6 and Table 7 of our paper's appendix. As the paper has been finalized and cannot be modified, we are providing this clarification and correction here in this repository to assist you in successfully reproducing our experimental results. 
>
> The hyperparameter configurations for experiments are available in the `./scripts` directory of this repository. We appreciate your understanding and support.

### Citation  

```BibTex
@inproceedings{he2025str,
  title={Str-GCL: Structural Commonsense Driven Graph Contrastive Learning},
  author={He, Dongxiao and Huang, Yongqi and Zhao, Jitao and Wang, Xiaobao and Wang, Zhen},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={1129--1141},
  year={2025}
}
```
