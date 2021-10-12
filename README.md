# Semi-Supervised Semantic Segmentation via Adaptive Equalization Learning, NeurIPS 2021 (Spotlight)

## Abstract
Due to the limited and even imbalanced data, semi-supervised semantic segmentation tends to have poor performance on some certain categories, e.g., tailed categories in Cityscapes dataset which exhibits a long-tailed label distribution. Existing approaches almost all neglect this problem, and treat categories equally. Some popular approaches such as consistency regularization or pseudo-labeling may even harm the learning of under-performing categories, that the predictions or pseudo labels of these categories could be too inaccurate to guide the learning on the unlabeled data. In this paper, we look into this problem, and propose a novel framework for semi-supervised semantic segmentation, named adaptive equalization learning (AEL). AEL adaptively balances the training of well and badly performed categories, with a confidence bank to dynamically track category-wise performance during training. The confidence bank is leveraged as an indicator to tilt training towards under-performing categories, instantiated in three strategies: 1) adaptive Copy-Paste and CutMix data augmentation approaches which give more chance for under-performing categories to be copied or cut; 2) an adaptive data sampling approach to encourage pixels from under-performing category to be sampled; 3) a simple yet effective re-weighting method to alleviate the training noise raised by pseudo-labeling. Experimentally, AEL outperforms the state-of-the-art methods by a large margin on the Cityscapes and Pascal VOC benchmarks under various data partition protocols. For more details, please refer to our NeurIPS paper ([arxiv](https://arxiv.org/pdf/2110.05474.pdf)). 

![image](https://github.com/hzhupku/SemiSeg-AEL/blob/main/arch.PNG)

## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions. 

## Training and Evaluation
For example, perform training and evaluation with 1/2 data parttition on Cityscapes dataset.
```bash
cd experiments/cityscapes_2
bash train.sh
```
For other partition protocols, change n_sup in config.yaml.
#### TODO
- [ ] Other SOTA semi-supervised segmentation methods
