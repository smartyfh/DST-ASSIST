# DST-STAR

The [STAR](https://arxiv.org/abs/2101.09374) model is another primary model we considered. It is an ontology-based model and has achieved the SOTA performance on MultiWOZ 2.4.

## Usage

### Data Preprocessing

There are two steps that preprocess the dataset to the format required by the model.

```console
❱❱❱ python3 create_data.py --mwz_ver 2.0
❱❱❱ python3 preprocess_data.py --data_dir data/mwz2.0
```

or

```console
❱❱❱ python3 create_data.py --mwz_ver 2.4
❱❱❱ python3 preprocess_data.py --data_dir data/mwz2.4
```

### Training

Before training the model, we need to copy the generated pseudo labels by the AUX-DST model to the current directory at first. Then simply run:

```console
❱❱❱ python3 train-pseudo.py --data_dir data/mwz2.0 --save_dir output-pseudo-20/exp --alpha 0.6 --do_train
```

or


```console
❱❱❱ python3 train-pseudo.py --data_dir data/mwz2.4 --save_dir output-pseudo-24/exp --alpha 0.4 --do_train
```

## As Auxiliary Model

We have also tested adopting STAR as the auxiliary model. 

```console
❱❱❱ python3 train-aux.py --data_dir data/mwz2.0 --save_dir output-star-20/exp --do_train
❱❱❱ python3 train-aux.py --data_dir data/mwz2.0 --save_dir output-star-20/exp
❱❱❱ mv pred/preds_1.json pseudo_label_star_20.json
```

or 

```console
❱❱❱ python3 train-aux.py --data_dir data/mwz2.4 --save_dir output-star-24/exp --do_train
❱❱❱ python3 train-aux.py --data_dir data/mwz2.4 --save_dir output-star-24/exp
❱❱❱ mv pred/preds_1.json pseudo_label_star_24.json
```

## Links

* The pseudp labels generated by AUX-DST can be downloaded [here](https://drive.google.com/file/d/1xrzhbEIou7h-qS1yRd83vKVnR6ZGmotp/view?usp=sharing).