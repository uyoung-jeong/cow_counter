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
Download the preprocessed pkl file from [this link](https://drive.google.com/file/d/1juJ9lj4eEtZgg4lgXqITuI2zqgHgAYps/view?usp=share_link).
In order to modify and make your own pkl file, you need to download [COCO](https://cocodataset.org/#home) 2017 train images and instance annotations.
In `prepro_data.py`, specify the coco directory and run the script.

## Directory hierarchy
```
{ROOT}
|── data
|   └── coco_cow_nfeatures_128.pkl
|── config
|   └── ...
|── lib
|   └── ...
```

## How to run
```shell
# run SVR
python main.py --cfg config/svr.yaml
```
