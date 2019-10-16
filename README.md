![logo](images/linclogo.png)

# LINC identification project

This project intends to help LINC in the process of identifying lions by processing their pictures through software. In particular this project contains a working face identifier and a prototype of a whisker identifier. 

The face id project was built using [fastai](fast.ai) and uses [LINC object detector](https://github.com/tryolabs/LINC) to preprocess images.

The whisker id baseline uses [Point Cloud Utils](https://github.com/fwilliams/point-cloud-utils) and applies *Approximate Wasserstein (Sinkhorn) Distance Between Point-Clouds*.

Face id example, `Original` shows the query image and then from left to right predictions are shown (ordered by confidence):
![logo](images/face_id_example.jpg)

## Installation
Python 3.6 or newer is needed.

First, clone this repository and run:
```bash
pip install -r requirements.txt
```
## Usage

### Creating datasets
There are two scripts for creating datasets under the `datasets` folder: `dataset_generator.py` and `train_val_split.py`. To use any of the scripts the dataset folder structure should be the following:
```
input/
      1/
        image_1.jpg
        image_2.jpg
        ...
      2/
        image_32.jpg
        ...
    ...
```
Note: The folder name must be the lion ID and images should contain `image_ID` on the filename (as shown above).

#### Dataset generator
This script creates a copy of a local dataset, algins and filters the images. 

Run `python dataset_generator.py --help` for usage info. 

#### Dataset splitter
This script splits the dataset into train and validation. Also filters lions that are beyond the minium amount of images threshold. The operation is __inplace__.

Run `python train_val_split.py --help` for usage info.

### Face identification

#### Training
The training script recives a dataset with the format specified before and creates a folder containing the trained model and the gallery needed for prediction.

Run `python train.py --help` for usage info.

#### Inference
The inference script needs the [model pickle](https://github.com/tryolabs/LINC/releases/download/v1.0/whiskers.pth) and the [gallery](https://github.com/tryolabs/LINC/releases/download/v1.0/body_parts.pth), these are created by the training script, or can be found on the [releases](https://github.com/tryolabs/LINC/releases) page of the repo.

The dataset that was used for training the model was created using the defaults from the dataset generator and split (zoomed images, filter side faces and a minimun of 3 images per lion).

Run `python predict.py --help` for usage info.

#### Adding new lions
Given a folder with the structure specified before, `add_new_lions.py` creates a new gallery that includes all the lions on the input folder. Also if some lions were already present on the dataset, all their new image ids will be added.

Run `python add_new_lions.py --help` for usage info.

### Notebooks
There are several jupyter notebooks in the `notebooks/` directory which are useful for data exploration, and model evaluation results exploration. There is also a demo notebook for running an inference step chaining both models.