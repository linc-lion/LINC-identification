import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fastai.callbacks import SaveModelCallback
from fastai.vision import (
    ImageDataBunch,
    accuracy,
    cnn_learner,
    get_transforms,
    imagenet_stats,
    models,
    top_k_accuracy,
)

from utils import get_embeddings, get_image_ids


def train(data_path, model_output_path, gallery_output_path=None):
    """Trains a ResNet 50 model using the images on data_path. Output model is
    saved on model_output_path. If gallery_output_path is provided, a new gallery
    with the lions will be created there.
    Parameters
    ----------
    data_path : str
        Path to the folder containing the labeled images. They should be
        separated into train and valid.
    model_output_path : str
        Path to the folder where the models will be saved.
    gallery_output_path : str, optional, default: None
        If provided a new gallery will be created on the given path
    """

    print("Loading data...")
    data_path = Path(data_path)
    model_output_path = Path(model_output_path) / datetime.today().strftime("%m-%d-%y")
    model_output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data = ImageDataBunch.from_folder(data_path, ds_tfms=get_transforms(), size=224).normalize(
        imagenet_stats
    )

    print("Creating model...")
    # Create model
    learn = cnn_learner(
        data, models.resnet50, metrics=[accuracy, top_k_accuracy], path=model_output_path
    )

    print("Strating training phase 1...")
    # Fit last layer
    learn.fit_one_cycle(
        10, callbacks=[SaveModelCallback(learn, monitor="accuracy", name="best_model_stg_1")]
    )
    learn.load("best_model_stg_1")

    print("Strating training phase 2...")
    # Unfreeze model and fit all layers
    learn.unfreeze()
    learn.fit_one_cycle(
        20, callbacks=[SaveModelCallback(learn, monitor="accuracy", name="best_model_stg_2")]
    )
    learn.load("best_model_stg_2")

    # Save inference model
    learn.export("model.pkl")

    if gallery_output_path:
        print("Creating gallery...")
        # Create gallery (embeddings, image ids and labels)
        gallery_path = Path(gallery_output_path) / datetime.today().strftime("%m-%d-%y")
        gallery_path.mkdir(exist_ok=True)
        fixed_dl = learn.data.train_dl.new(shuffle=False, drop_last=False)

        embeddings = get_embeddings(learn, fixed_dl, pool=None)
        image_ids = get_image_ids(data_path / "train")

        labels = np.array(learn.data.train_ds.y.classes)[learn.data.train_ds.y.items]
        labels = [int(label) for label in labels]

        torch.save(embeddings, gallery_path / "embeddings.pt")
        torch.save(labels, gallery_path / "labels.pt")
        torch.save(image_ids, gallery_path / "face_image_ids.pt")
    print("Finished!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC face training")
    parser.add_argument("data_path", help="Path to the folder containing the labeled images")
    parser.add_argument(
        "model_output_path", help="Path to the folder where the models will be saved"
    )
    parser.add_argument(
        "--gallery_output_path",
        default=None,
        help="If provided a new gallery will be created on the given path.",
    )
    args = parser.parse_args()

    tic = time.time()
    train(args.data_path, args.model_output_path, args.gallery_output_path)
    toc = time.time()
    print(f"Done in {toc - tic:.2f} seconds!")
