# High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation
Pytorch Implementation of our paper "High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation"[[arXiv]](https://arxiv.org/abs/2204.11448).

More details can be found at the [homepage](https://njuvision.github.io/TinyLIC/). 

## News
- The latest version of our TinyLIC is released with more efficient network architecture in both transform and entropy coding modules. More details can be found in the paper.
- This project is still being updated. 

## Installation
To get started locally and install the development version of our work, run the following commands (The [docker environment](https://hub.docker.com/layers/pytorch/pytorch/1.11.0-cuda11.3-cudnn8-devel/images/sha256-9bfcfa72b6b244c1fbfa24864eec97fb29cfafc065999e9a9ba913fa1e690a02?context=explore) is recommended):
```bash
git clone https://github.com/lumingzzz/TinyLIC.git
cd TinyLIC
pip install -U pip && pip install -e .
```

## Usage

### Train
We use the [Flicker2W](https://github.com/liujiaheng/CompressionData) dataset for training, and the [script](https://github.com/xyq7/InvCompress/tree/main/codes/scripts) for preprocessing.

Run the script for a simple training pipeline:
```bash
python examples/train.py -m tinylic -d /path/to/my/image/dataset/ --epochs 400 -lr 1e-4 --batch-size 8 --cuda --save
```
The training checkpoints will be generated in the "pretrained" folder at current directory. You can change the default folder by modifying the function "init" in "./expample/train.py".


### Evaluation
Pre-trained models can be downloaded in [NJU Box](https://box.nju.edu.cn/d/6bd0aafa2faf47cab7c2/).

The mse optimized results can be found in [/results](https://github.com/lumingzzz/TinyLIC/tree/main/results) for reference.

An example to evaluate model:
```bash
python -m compressai.utils.eval_model checkpoint path/to/eval/data/ -a tinylic -p path/to/pretrained/model --cuda
```

## Citation
If you find this work useful for your research, please cite:

```
@article{lu2022high,
  title={High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation},
  author={Lu, Ming and Ma, Zhan},
  journal={arXiv preprint arXiv:2204.11448},
  year={2022}
}
```

## Acknowledgement
The framework is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI/), we add our modifications in compressai.models.tinylic and compressai.layers for usage.

The TinyLIC model is partially built upon the [Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer) and the open sourced unofficial implementation of [checkerboard shaped context model](https://github.com/leelitian/Checkerboard-Context-Model-Pytorch). We thank the authors for sharing their code.

## Contact
If you have any question, please contact me via luming@smail.nju.edu.cn.
