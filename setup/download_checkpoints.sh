# Script to download the checkpoints used in the paper
ckpt_root=$1
if [ -z "$ckpt_root" ]; then
    echo "Please specify the checkpoint root directory"
    echo "Usage: bash setup/download_checkpoints.sh <ckpt_root>"
    echo "For example: bash setup/download_checkpoints.sh /home/user/checkpoints"
    exit 1
fi


echo ":::: Saving all checkpoints to $ckpt_root ::::"
mkdir -p $ckpt_root
cd $ckpt_root

base_url="https://isis-data.science.uva.nl/testoftime/checkpoints/"
checkpoints=(
    "tempo-hparams_1.0_1.0_1.0-epoch=27-step=8288.ckpt"
    "charadesego-hparams_1.0_1.0_1.0-epoch=2-step=3639.ckpt"
    "charades-hparams_1.0_1.0_0.0-epoch=3-step=3120.ckpt"
    "activitynet-hparams_1.0_1.0_0.0-epoch=9-step=7450.ckpt"
)
for ckpt in "${checkpoints[@]}"; do
    echo ":::: Downloading $ckpt ::::"
    wget "$base_url/$ckpt"
done

cd -
