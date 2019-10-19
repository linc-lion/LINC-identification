import time
from pathlib import Path

import numpy as np
import torch
from fastai.vision import ImageList, imagenet_stats, load_learner
from sklearn.neighbors import KNeighborsClassifier

from utils import get_embeddings, get_k_neighbors


def get_neighbors(nearest_neighbors, nearest_distances, probe_ids, n):
    k_neighbors = {}
    for probe_id, neighbors, distances in zip(probe_ids, nearest_neighbors, nearest_distances):
        k_neighbors[probe_id] = get_k_neighbors(neighbors, distances, n)
    return k_neighbors


def get_top_n(emb_gal, label_gal, emb_probes, probe_ids, lion_subset, n):

    emb_subset = emb_gal[np.isin(label_gal, lion_subset)]
    label_subset = label_gal[np.isin(label_gal, lion_subset)]

    knn_classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn_classifier.fit(emb_subset, label_subset)

    nearest_neighbors = knn_classifier.kneighbors(emb_probes, n_neighbors=20, return_distance=True)

    topN = get_neighbors(label_subset[nearest_neighbors[1]], nearest_neighbors[0], probe_ids, n)

    return topN


def predict(query_image_set_path, n, model_path, gallery_path, lion_subset=None):

    # Load the model
    model_path = Path(model_path)
    learn = load_learner(model_path.parent, model_path.name)

    images_path = Path(query_image_set_path)
    gallery_path = Path(gallery_path)

    # Load the database
    disk_embeddings = torch.load(gallery_path / "embeddings.pt")
    disk_labels = torch.load(gallery_path / "labels.pt")

    if lion_subset is None:
        # If None search over all lions.
        lion_subset = disk_labels

    # Load incoming images
    incoming_data = (
        ImageList.from_folder(images_path)
        .split_none()
        .label_from_re(r"image_(\d*)")
        .transform(None, size=224)
        .databunch(bs=1)
        .normalize(imagenet_stats)
    )

    # Get embeddings from incoming images
    fixed_dl = incoming_data.train_dl.new(shuffle=False, drop_last=False)
    incoming_embeddings = get_embeddings(learn, fixed_dl, pool=None)

    incoming_ids = np.array([int(image_id) for image_id in incoming_data.train_ds.y.classes])

    # Get predictions
    return get_top_n(
        disk_embeddings,
        disk_labels,
        incoming_embeddings,
        incoming_ids[incoming_data.train_ds.y.items],
        lion_subset,
        n,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC Face Prediction")
    parser.add_argument(
        "query_image_set_path", help="Path to the folder containing the labeled images"
    )
    parser.add_argument("n", default=3, help="How many lions to retrive per image.")
    parser.add_argument("model_path", help="Path to the pickle of the model")
    parser.add_argument(
        "gallery_path",
        help="Path to the folder containing the gallery: embeddings.pt, image_ids.pt and labels.pt",
    )
    parser.add_argument(
        "--lion_subset",
        default=None,
        help="Comma separated list of the lion ids to be matched agianst. Default: All lions",
    )

    args = parser.parse_args()

    lion_subset = (
        [int(lion_id) for lion_id in args.lion_subset.split(",")] if args.lion_subset else None
    )

    tic = time.time()
    results = predict(
        args.query_image_set_path, float(args.n), args.model_path, args.gallery_path, lion_subset
    )
    toc = time.time()
    print(results, f"Done in {toc - tic:.2f} seconds!")
