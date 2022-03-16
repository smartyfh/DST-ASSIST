# AUX-DST

This model is used as both the auxiliary model and the primary model. When used as the auxiliary model, it is trained on the clean dev set and we chose the best checkpoint according to the performance on the noisy training set. When used as the primary model, it is trained on the noisy training set and the best model was chosen based on the performance on the dev set. The model was tested on MultiWOZ 2.0 (*a modified version of the original one*) and MultiWOZ 2.4.

## Usage

Here I show how we can run this model on MultiWOZ 2.0. The same procedure can be applied to MultiWOZ 2.4.

### Data Preprocessing

There are two steps that preprocess the dataset to the format required by the model.

```console
❱❱❱ python3 create_data.py --mwz_ver 2.0
❱❱❱ python3 preprocess_data.py --data_dir data/mwz2.0
```

### As Auxiliary Model

After preprocessing the dataset, we first train the auxiliary model and then generate pseudo labels.

```console
❱❱❱ python3 train-aux.py --data_dir data/mwz2.0 --save_dir output-aux-20/exp --do_train
❱❱❱ python3 train-aux.py --data_dir data/mwz2.0 --save_dir output-aux-20/exp
❱❱❱ mv pred/preds_1.json pseudo_label_aux_20.json
```

### As Primary Model

After obtaining the pseudo labels, we train the primary model using both the original noisy labels and the generated pseudo labels.

```console
❱❱❱ python3 train-pseudo.py --data_dir data/mwz2.0 --save_dir output-pseudo-20/exp --alpha 0.6 --do_train
```


## Links

* The generated pseudp labels can be downloaded [here](https://drive.google.com/file/d/1xrzhbEIou7h-qS1yRd83vKVnR6ZGmotp/view?usp=sharing).

* The model checkpoints can be downloaded here: [2.0](https://drive.google.com/file/d/1QfwzHLWbWJh9pdJ5S1uonk8i6bUgZQdC/view?usp=sharing), [2.4](https://drive.google.com/file/d/1TLvozE_ezfiMHDiisTJVYUuJ93_h9SIp/view?usp=sharing).
