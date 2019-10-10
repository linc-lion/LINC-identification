from pathlib import Path

import numpy as np
import torch
from fastai.vision import ImageList, imagenet_stats, load_learner

from config import configuration
from utils import get_embeddings, get_image_ids


def add_new_lions(new_lions_path, output_gallery_path):

    # Load the model
    learn = load_learner(configuration["FACE_MODEL_PATH"])

    images_path = Path(new_lions_path)

    output_path = Path(output_gallery_path) / "gallery"
    gallery_path = Path(configuration["GALLERY_PATH"])

    # Load the database
    disk_embeddings = torch.load(gallery_path / "embeddings.pt")
    disk_image_ids = torch.load(gallery_path / "image_ids.pt")
    disk_labels = torch.load(gallery_path / "labels.pt")

    # Load incoming images
    incoming_data = (
        ImageList.from_folder(images_path)
        .split_none()
        .label_from_folder()
        .transform(None, size=224)
        .databunch()
        .normalize(imagenet_stats)
    )

    # Get embeddings from incoming images
    fixed_dl = incoming_data.train_dl.new(shuffle=False, drop_last=False)
    incoming_embeddings = get_embeddings(learn, fixed_dl, pool=None)
    incoming_labels = np.array(incoming_data.train_ds.y.classes)[incoming_data.train_ds.y.items]

    new_embeddings = torch.cat((disk_embeddings, incoming_embeddings))
    new_image_ids = get_image_ids(images_path, disk_image_ids)
    new_labels = np.concatenate((disk_labels, incoming_labels))

    output_path.mkdir(exist_ok=True)

    torch.save(new_embeddings, output_path / "embeddings.pt")
    torch.save(new_image_ids, output_path / "image_ids.pt")
    torch.save(new_labels, output_path / "labels.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC add new lions")
    parser.add_argument("new_lions_path", help="Path to the folder containing the labeled images")
    parser.add_argument(
        "output_gallery_path",
        help="Path to the destination where the new gallery is going to be created",
    )
    args = parser.parse_args()

    add_new_lions(args.new_lions_path)
