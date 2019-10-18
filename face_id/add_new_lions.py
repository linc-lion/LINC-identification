from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from fastai.vision import ImageList, imagenet_stats, load_learner

from utils import get_embeddings, get_image_ids


def add_new_lions(new_lions_path, output_path, model_path, gallery_path):

    # Load the model
    print("Loading model...")
    model_path = Path(model_path)
    learn = load_learner(model_path.parent, model_path.name)

    images_path = Path(new_lions_path)

    output_path = Path(output_path) / "gallery-{}".format(datetime.today().strftime("%m-%d-%y"))
    output_path.mkdir()

    gallery_path = Path(gallery_path)

    # Load the database
    print("Loading gallery...")
    disk_embeddings = torch.load(gallery_path / "embeddings.pt")
    disk_image_ids = torch.load(gallery_path / "image_ids.pt")
    disk_labels = torch.load(gallery_path / "labels.pt")

    # Load incoming images
    incoming_data = (
        ImageList.from_folder(images_path)
        .split_none()
        .label_from_folder()
        .transform(None, size=224)
        .databunch(bs=1)
        .normalize(imagenet_stats)
    )

    # Get embeddings from incoming images
    print("Getting embeddings...")
    fixed_dl = incoming_data.train_dl.new(shuffle=False, drop_last=False)
    incoming_embeddings = get_embeddings(learn, fixed_dl, pool=None)
    incoming_labels = np.array(incoming_data.train_ds.y.classes)[incoming_data.train_ds.y.items]

    new_embeddings = torch.cat((disk_embeddings, incoming_embeddings))
    new_image_ids = get_image_ids(images_path, disk_image_ids)
    new_labels = np.concatenate((disk_labels, incoming_labels))

    torch.save(new_embeddings, output_path / "embeddings.pt")
    torch.save(new_image_ids, output_path / "image_ids.pt")
    torch.save(new_labels, output_path / "labels.pt")

    print(f"Lions ${incoming_data.train_ds.y.classes} successfully added/updated!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC add new lions")
    parser.add_argument("new_lions_path", help="Path to the folder containing the labeled images")
    parser.add_argument(
        "output_path", help="Path to the destination where the new gallery is going to be created"
    )
    parser.add_argument("model_path", help="Path to the pickle of the model")
    parser.add_argument(
        "gallery_path",
        help="Path to the folder containing the gallery: embeddings.pt, image_ids.pt and labels.pt",
    )
    args = parser.parse_args()

    add_new_lions(args.new_lions_path, args.output_path, args.model_path, args.gallery_path)
