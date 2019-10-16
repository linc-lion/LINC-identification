from functools import partial
from pathlib import Path

import torch
from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback
from fastai.torch_core import defaults
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


def train(data_path, output_path, create_gallery=True, force_cpu=False):

    defaults.device = (
        torch.device("cuda") if torch.cuda.is_available() and not force_cpu else torch.device("cpu")
    )

    data_path = Path(data_path)
    output_path = Path(output_path)
    model_path = output_path / "model"
    model_path.mkdir()
    gallery_path = output_path / "gallery"
    gallery_path.mkdir()

    # Load data
    data = ImageDataBunch.from_folder(data_path, ds_tfms=get_transforms(), size=224).normalize(
        imagenet_stats
    )

    # Create model
    learn = cnn_learner(
        data,
        models.resnet50,
        metrics=[accuracy, top_k_accuracy],
        callback_fns=[
            partial(EarlyStoppingCallback, monitor="accuracy", min_delta=0.005, patience=5)
        ],
        path=model_path,
    )

    # Fit last layer
    learn.fit_one_cycle(
        10, callbacks=[SaveModelCallback(learn, monitor="accuracy", name="best_model_stg_1")]
    )
    learn.load("best_model_stg_1")

    # Unfreeze model and fit all layers
    learn.unfreeze()
    learn.fit_one_cycle(
        20, callbacks=[SaveModelCallback(learn, monitor="accuracy", name="best_model_stg_2")]
    )
    learn.load("best_model_stg_2")

    # Save inference model
    learn.export("model.pkl")

    if create_gallery:
        # Create gallery (embeddings, image ids and labels)
        fixed_dl = learn.train_dl.new(shuffle=False, drop_last=False)

        embeddings = get_embeddings(learn, fixed_dl, pool=None)
        image_ids = get_image_ids(data_path)

        torch.save(embeddings, gallery_path / "embeddings.pt")
        torch.save(learn.train_ds.y.items, gallery_path / "labels.pt")
        torch.save(image_ids, gallery_path / "image_ids.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC Face Prediction")
    parser.add_argument("data_path", help="Path to the folder containing the labeled images")
    parser.add_argument(
        "output_path", help="Path to the folder where the models and gallery will be saved"
    )
    parser.add_argument("--cpu", dest="cpu", help="Force model to use CPU", action="store_true")
    parser.add_argument(
        "--no_gallery",
        default=False,
        dest="no_gallery",
        help="Stop the creation of a new gallery",
        action="store_true",
    )
    args = parser.parse_args()

    results = train(args.data_path, args.output_path, not args.no_gallery, args.cpu)

    print(results)