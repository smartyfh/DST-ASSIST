# DST-ASSIST

This is the implementation of our work: **ASSIST: Towards Label Noise-Robust Dialogue State Tracking. Fanghua Ye, Yue Feng, Emine Yilmaz. Findings of ACL 2022.** [[paper](https://arxiv.org/abs/2202.13024)]

## Abstract

The MultiWOZ 2.0 dataset has greatly boosted the research on dialogue state tracking (DST). However, substantial noise has been discovered in its state annotations. Such noise brings about huge challenges for training DST models robustly. Although several refined versions, including MultiWOZ 2.1-2.4, have been published recently, there are still lots of noisy labels, especially in the training set. Besides, it is costly to rectify all the problematic annotations. In this paper, instead of improving the annotation quality further, we propose a general framework, named ASSIST (lAbel noiSe-robuSt dIalogue State Tracking), to train DST models robustly from noisy labels. ASSIST first generates pseudo labels for each sample in the training set by using an auxiliary model trained on a small clean dataset, then puts the generated pseudo labels and vanilla noisy labels together to train the primary model. We show the validity of ASSIST theoretically. Experimental results also demonstrate that ASSIST improves the joint goal accuracy of DST by up to 28.16% on MultiWOZ 2.0 and 8.41% on MultiWOZ 2.4, compared to using only the vanilla noisy labels.

## Usage

Please refer to each method for the details.

+ [AUX-DST](https://github.com/smartyfh/DST-ASSIST/tree/main/AUX-DST)
+ [STAR](https://github.com/smartyfh/DST-ASSIST/tree/main/STAR)
+ [SOM-DST](https://github.com/smartyfh/DST-ASSIST/tree/main/SOM-DST)

## Citation

```bibtex
@inproceedings{ye2022assist,
  title={ASSIST: Towards Label Noise-Robust Dialogue State Tracking},
  author={Ye, Fanghua and Feng, Yue and Yilmaz, Emine},
  journal={arXiv preprint arXiv:2202.13024},
  year={2022}
  }
```
