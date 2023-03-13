## Setting up a development environment

If you do not have `conda` installed, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html). This code has been tested on Linux.

Create a new `conda` environment with Python 3.8.8:

```bash
conda create -n testoftime python=3.8.8 -y
```

Install the required packages:

0. Activate the environment:
```bash
conda activate testoftime
```

1. Install PyTorch 1.12.1 with CUDA 11.1 support:

```sh
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Depending on your CUDA version, you may have to install this differently. See [here](https://pytorch.org/get-started/previous-versions/) for more details.


Note: Before proceeding, make sure to add `$PWD` to the `PYTHONPATH` environment variable:

```sh
export PYTHONPATH=$PWD:$PYTHONPATH
```


2. Setup code for `VideoCLIP` model: We borrow code for VideoCLIP model from [the official repository](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT). In this repo, we provide an adaptation of their code in `external/fairseq/examples/MMPT`. The credit for the original code solely lies with the authors of the [original repository](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT) at FAIR.


    2.1: Download best VideoCLIP checkpoint:
    ```sh
    ckpt_dir=external/requirements/fairseq/runs/retri/videoclip
    mkdir -p $ckpt_dir
    wget https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt -O $ckpt_dir/checkpoint_best.pt
    ```

    <!-- 2.2: Download S3D model checkpoints.
    ```sh
    ckpt_dir=external/requirements/fairseq/pretrained_models/
    mkdir -p $ckpt_dir
    wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth -O $ckpt_dir/s3d_howto100m.pth
    wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy -O $ckpt_dir/s3d_dict.npy
    ``` -->

    2.2: Check that loading the model works:
    ```sh
    python external/utils_videoclip.py
    ```


2. Install the rest of the required packages (in this order):

```sh
pip install ipdb
pip install transformers==3.4
pip install protobuf==3.20.0
pip install omegaconf==2.0.6
pip install termcolor==2.2.0
pip install pytorch_lightning==1.7.6
pip install wandb==0.13.3
pip install editdistance==0.6.2
pip install pandas==1.5.3
```

3. Check that the installation was successful:

```sh
python setup/check_packages.py
```