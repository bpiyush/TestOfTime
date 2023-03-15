##  ⏰ [Test of Time: Instilling Video-Language Models with a Sense of Time](https://arxiv.org/abs/2301.02074)

Code for our CVPR 2023 [paper](https://arxiv.org/abs/2301.02074) on instilling a sense of time in video-language models.

## Installation & Setup

Create a `conda` environment and install packages as described in [`setup/env.md`](setup/env.md).

## Overview

We present <u>T</u>emporal <u>A</u>daptation by <u>C</u>onsistent <u>T</u>ime-ordering (TACT) as a way of making video-language models understand before/after relations in text and connecting them with pair of events in a video stream.

## Datasets

We use a combination of synthetic and real datasets to evaluate our approach. Below, you can find instructions to download and prepare the datasets. Here, we present instructions for our Synthetic dataset and the [TEMPO-TL](https://arxiv.org/abs/1809.01337v1) dataset.

### Synthetic data

We create simple synthetic video-language pairs by stitching together a pair of events (e.g., "a <span style="color:red">red</span> circle appears" and "a <span style="color:yellow">yellow</span> circle appears") with text description connected by *before/after* relations. An example is shown here:

![Synthetic data](media/synthetic-data-v3.gif)

TODO: Add instructions to download.

### TEMPO-TL dataset

As a real dataset, we consider the [TEMPO-TL](https://arxiv.org/abs/1809.01337v1) dataset that similarly stitches together a pair of events in text for clips in the same video.

![TEMPO-TL data](media/tempo-data-v1.gif)

TODO

### Other datasets

In order to evaluate our approach on other datasets, you need to first generate and save S3D video features. Then, create splits, create a dataset object in `package/datasets/`. Please see `package/datasets/tempo.py` for reference.

## Models

We base our experiments on the VideoCLIP model from FAIR. Instructions in [`setup/env.md`](setup/env.md) include download of relevant checkpoints for VideoCLIP.

## Checkpoints
TODO

## Post-pretraining: TACT

* Post-pretraining on TEMPO-TL dataset
    ```sh
    python postpretrain.py --dataset tempo --eval_subset temporal_1k --no_wandb --data_root /ssd/pbagad/datasets/ --only_train
    ```
    Replace `--data_root` with the path to where all your dataseta are stored. Make sure to change `entity` and `project` arguments in [`postpretrain.py`](postpretrain.py) to log to your own wandb account.

## Evaluation: TACT

#### Evaluate on `TEMPO` dataset

* Pre-trained VideoCLIP
    ```sh
    python postpretrain.py --dataset tempo --eval_subset temporal_1k --eval_split test --only_eval --no_wandb --data_root /ssd/pbagad/datasets/
    ```
    Replace `--data_root` with the path to where all your dataseta are stored. This should yield about 52% accuracy.

* TACT post-pretrained VideoCLIP
    ```sh
    ckpt=/path/to/tact/checkpoint/trained/on/TEMPO/
    # For example, ckpt=test-of-time/1arb5f3m/checkpoints/epoch=27-step=8288.ckpt
    python postpretrain.py --dataset tempo --eval_subset temporal_1k --eval_split test --only_eval --no_wandb --data_root /ssd/pbagad/datasets/ -c $ckpt
    ```
    Replace `--data_root` with the path to where all your dataseta are stored. This should yield about 66% accuracy.

#### Evaluate on `Synthetic` dataset

* Pre-trained VideoCLIP
    ```sh
    python postpretrain.py --dataset synthetic --eval_subset v2.0 --eval_split test --only_eval --no_wandb --data_root /ssd/pbagad/datasets/
    ```
    Replace `--data_root` with the path to where all your dataseta are stored. This should yield about 45% accuracy.

* TACT post-pretrained VideoCLIP
    ```sh
    ckpt=/path/to/tact/checkpoint/trained/on/TEMPO/
    # For example, ckpt=test-of-time/1arb5f3m/checkpoints/epoch=27-step=8288.ckpt
    python postpretrain.py --dataset synthetic --eval_subset v2.0 --eval_split test --only_eval --no_wandb --data_root /ssd/pbagad/datasets/ -c $ckpt
    ```
    Replace `--data_root` with the path to where all your dataseta are stored. This should yield about 78% accuracy.


## Evaluation: Downstream Tasks
TODO

## Citation

If you found our work useful or relevant, please consider citing our paper:

```bibtex
@inproceedings{
      bagad2023testoftime,
      title={{T}est of {T}ime: {I}nstilling {V}ideo-{L}anguage {M}odels with a {S}ense of {T}ime},
      author={Bagad, Piyush and Tapaswi, Makarand and Snoek, Cees G. M.},
      booktitle={CVPR},
      year={2023}
}
```

## Acknowledgements

We acknowledge support from the [ELLIS Amsterdam Unit](https://ivi.fnwi.uva.nl/ellis/) and the [AMS Scholarhsip](https://www.uva.nl/en/education/fees-and-funding/masters-scholarships-and-loans/faculty-scholarships-science/science.html) to Piyush as a Master's student.
We also acknowledge all relevent prior work, particularly, [VideoCLIP](https://arxiv.org/abs/2109.14084) and [TEMPO](https://arxiv.org/abs/1809.01337v1), for making their code and data publicly available.

### Additional Notes

> :warning: **Infra note**: Our code has been run on a single node with 4 GPUs (either NVIDIA RTX A5000 or NVIDIA GeForce 1080). Running it on different infrastructures may cause differences in results. However, the trends and inferences should be similar (e.g., post-pretraining helps with temporal ordering task, etc.).
