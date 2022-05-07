# High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation
Pytorch Implementation of our paper "High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation"[[arXiv]](https://arxiv.org/abs/2204.11448).

More details can be found at the [homepage](https://njuvision.github.io/TinyLIC/). 

## Acknowledgement
The framework is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI/), we add our networks in compressai.models.tic and compressai.layers for usage.

## Installation
To get started locally and install the development version of our work, run the following commands (The [docker environment](https://registry.hub.docker.com/layers/pytorch/pytorch/1.8.1-cuda11.1-cudnn8-devel/images/sha256-024af183411f136373a83f9a0e5d1a02fb11acb1b52fdcf4d73601912d0f09b1?context=explore) is recommended):
```bash
git clone https://github.com/lumingzzz/TIC.git
cd TIC
pip install -U pip && pip install -e .
pip install timm
```

## Usage

### Train
We use the [Flicker2W](https://github.com/liujiaheng/CompressionData) dataset for training, and the [script](https://github.com/xyq7/InvCompress/tree/main/codes/scripts) for preprocessing.

Run the script for a simple training pipeline:
```bash
python examples/train.py -m tinylic -d /path/to/my/image/dataset/ --epochs 400 -lr 1e-4 --batch-size 8 --cuda --save
```


<!-- ### Evaluation
A pre-trained model of Quality 3 (lambda = 0.0067) can be downloaded in [NJU Box](https://box.nju.edu.cn/f/63e702b5997c4572a2e6/?dl=1). Since the codes are reorgnized and retrained using the newest version (1.2.0) of [CompressAI](https://github.com/InterDigitalInc/CompressAI/), the performance (0.286/31.62) may be a little deviation from that in the paper (0.281/31.50). -->

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
## Contact
If you have any question, please contact me via luming@smail.nju.edu.cn.

## Related links
 * Kodak image dataset: http://r0k.us/graphics/kodak/
 * BPG image format by _Fabrice Bellard_: https://bellard.org/bpg
 * VVC VTM reference software: https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM
