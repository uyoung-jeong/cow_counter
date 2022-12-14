# Cow Counter
Regress the number of cows in an image.

## Installation
### Setup conda environment
```shell
conda create -n cow python=3.8
conda activate cow
```

### Install python libraries
```shell
pip install -r requirements.txt
```
### Dataset setup
Download the preprocessed pkl file from [this link](https://drive.google.com/drive/folders/1yR0O5AT4sZj0KC2x208bAi_xwh4S-Igo?usp=sharing).
In order to modify and make your own pkl file, you need to download [COCO](https://cocodataset.org/#home) 2017 train images and instance annotations.
In `prepro_data.py`, specify the coco directory and run the script.

## Directory hierarchy
```
{ROOT}
|── data
|   |── coco_cow_nfeatures_128.pkl
    └── ...
|── config
|   └── ...
|── lib
|   └── ...
```

## How to run
```shell
# run SVR
python main.py --cfg config/svr.yaml
# run MLP
python main.py --cfg config/mlp.yaml
```
