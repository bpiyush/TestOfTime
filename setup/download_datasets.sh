# Script to download the datasets used in the paper
data_root=$1

if [ -z "$data_root" ]; then
    echo "Please specify the data root directory"
    echo "Usage: bash setup/download_datasets.sh <data_root>"
    echo "For example: bash setup/download_datasets.sh /home/user/data"
    exit 1
fi

echo ":::: Saving all datasets to $data_root ::::"
mkdir -p $data_root
cd $data_root

base_url=https://isis-data.science.uva.nl/testoftime/datasets


dataset="ToT-syn-v2.0.zip"
echo ":::: Downloading $dataset ::::"
wget "$base_url/$dataset"
unzip $dataset

dataset="agqa.zip"
echo ":::: Downloading $dataset ::::"
wget "$base_url/$dataset"
unzip $dataset

dataset="charades.zip"
echo ":::: Downloading $dataset ::::"
wget "$base_url/$dataset"
unzip $dataset

dataset="didemo.zip"
echo ":::: Downloading $dataset ::::"
wget "$base_url/$dataset"
unzip $dataset

dataset="tempo.zip"
echo ":::: Downloading $dataset ::::"
wget "$base_url/$dataset"
unzip $dataset

dataset="ssv2.zip"
echo ":::: Downloading $dataset ::::"
wget "$base_url/$dataset"
unzip $dataset

cd -
