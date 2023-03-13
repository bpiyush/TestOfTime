##  ⏰ [Test of Time: Instilling Video-Language Models with a Sense of Time](https://arxiv.org/abs/2301.02074)

Code for our CVPR 2023 [paper](https://arxiv.org/abs/2301.02074) on instilling a sense of time in video-language models.

## Installation & Setup

Create a `conda` environment and install packages as described in [`setup/env.md`](setup/env.md).


## Datasets

### Synthetic data

TODO

### TEMPO-TL dataset

TODO

## Models

We base our experiments on the VideoCLIP model from FAIR. Instructions in [`setup/env.md`](setup/env.md) include download of relevant checkpoints for VideoCLIP.

## Checkpoints

## Evaluation: Time-awareness

#### Evaluate on `TEMPO` dataset

* Pre-trained VideoCLIP
    ```sh
    python postpretrain.py --dataset tempo --subset temporal_1k --split test --only_eval --no_wandb --data_root /ssd/pbagad/datasets/
    ```
    Replace `--data_root` with the path to where all your dataseta are stored. This should yield about 52% accuracy.

* TACT post-pretrained VideoCLIP
    ```sh
    ckpt=/path/to/tact/checkpoint/trained/on/TEMPO/
    # For example, ckpt=test-of-time/1arb5f3m/checkpoints/epoch=27-step=8288.ckpt
    python postpretrain.py --dataset tempo --subset temporal_1k --split test --only_eval --no_wandb --data_root /ssd/pbagad/datasets/ -c $ckpt
    ```
    Replace `--data_root` with the path to where all your dataseta are stored. This should yield about 66% accuracy.

#### Evaluate on `Synthetic` dataset

* Pre-trained VideoCLIP
    ```sh
    python postpretrain.py --dataset synthetic --subset v2.0 --split test --only_eval --no_wandb --data_root /ssd/pbagad/datasets/
    ```
    Replace `--data_root` with the path to where all your dataseta are stored.

* TACT post-pretrained VideoCLIP
    ```sh
    ckpt=/path/to/tact/checkpoint/trained/on/TEMPO/
    # For example, ckpt=test-of-time/1arb5f3m/checkpoints/epoch=27-step=8288.ckpt
    python postpretrain.py --dataset synthetic --subset v2.0 --split test --only_eval --no_wandb --data_root /ssd/pbagad/datasets/ -c $ckpt
    ```
    Replace `--data_root` with the path to where all your dataseta are stored.

## Post-pretraining: <u>T</u>emporal <u>A</u>daptation by <u>C</u>onsistent <u>T</u>ime-ordering (TACT)

* Post-pretraining on TEMPO-TL dataset
    ```sh
    python postpretrain.py --dataset tempo --subset temporal_1k --no_wandb --data_root /ssd/pbagad/datasets/ --only_train
    ```
    Replace `--data_root` with the path to where all your dataseta are stored. Make sure to change `entity` and `project` arguments in [`postpretrain.py`](postpretrain.py) to log to your own wandb account.


**Infra note**: Our code has been run on a single node with 4 GPUs (either NVIDIA RTX A5000 or NVIDIA GeForce 1080). Running it on different infrastructures may cause slight differences in results.
