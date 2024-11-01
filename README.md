# HikingTTE

## Overview
HikingTTE is a deep learning model designed to predict the travel time estimation for hiking.

## Directory Structure
```
- Datasets
- models
    - HikingTTE
    - base
        - __init__.py
        - Attr.py
        - LSTMAttention.py
    - __init__.py
    - HikingTTE.py
- result
- src
    - common
        - data_loader_split.py
        - logger.py
        - utils.py
    - HikingTTE
        - data_attributes.json
        - main.py
    - data_preprocess
        - add_terrain_slope.ipynb
- environment.yml
```

## Environment Setup
**Required Libraries**

The required libraries are listed in environment.yml. Please set up the environment using the following commands:

```bash
conda env create -f environment.yml
conda activate your_env_name
```

## Preparing the Dataset
The dataset can be obtained from the following links.

https://www.kaggle.com/datasets/roccoli/gpx-hike-tracks

The dataset has been converted from GPX format to JSONL format and edited according to the procedures in the paper. The edited dataset is available from:

https://drive.google.com/drive/folders/1VOfGSdYJqLHeEXRhyZ2JVSBbFCfZBy9G?usp=sharing

Please place the following files in the `Datasets/hikr_org_train_test_valid` directory:
- `config.json`
- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`

Please note that the information for `terrain_slope` was obtained using NASADEM and ArcticDEM data. You can add this information using the following file:

```
src/data_preprocess/add_terrain_slope.ipynb
```

The dataset that includes the added `terrain_slope` information will be saved in `Datasets/processed_dataset`.


## Training and Evaluation of the Model

By running `src/HikingTTE/main.py`, you can train and evaluate the model.

**Training**

Start training the model with the following command:
```bash
python main.py --task train
```
If you do not wish to use ```WandB```, add the ```--no-wandb``` option:
```bash
python main.py --task train --no-wandb
```
You can specify the following parameters via command-line arguments:

- `--data_dir`: Directory of the dataset (default: Datasets/processed_dataset)
- `--batch_size`: Batch size (default: 64)
- `--num_epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)

For more details, refer to the `argparse` settings in `main.py`.

**Resuming Training**

To resume training from a checkpoint, use the following command:
```bash
python main.py --task resume --num_epochs 1000
```

Replace `--num epochs 1000` with the total number of epochs you wish to train.

**Evaluation**

To evaluate the model using a trained model, use the following command:

```bash
python main.py --task test
```
The evaluation results will be saved in the `result` directory.

## References
[1] GPS recorded hikes from hikr.org: [https://www.kaggle.com/datasets/roccoli/gpx-hike-tracks](https://www.kaggle.com/datasets/roccoli/gpx-hike-tracks)

[2] NASADEM: [https://doi.org/10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001](https://doi.org/10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001)

[3] ArcticDEM: [https://www.pgc.umn.edu/data/arcticdem/](https://www.pgc.umn.edu/data/arcticdem/)