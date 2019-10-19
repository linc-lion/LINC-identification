import time
from datetime import datetime
from pathlib import Path

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

    print("Loading data...")
    data_path = Path(data_path)
    model_output_path = Path(model_output_path) / datetime.today().strftime("%m-%d-%y")
    model_output_path.mkdir(parents=True)

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
        image_ids = get_image_ids(data_path)

        torch.save(embeddings, gallery_path / "embeddings.pt")
        torch.save(learn.data.train_ds.y.items, gallery_path / "labels.pt")
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
