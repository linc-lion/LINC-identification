import time
from pathlib import Path

import numpy as np
import point_cloud_utils as pcu
import torch
from PIL import Image
from pycpd import deformable_registration
from tqdm import tqdm

from utils import get_k_neighbors, load_model, whisker_detector_predict


@torch.no_grad()
def get_whisker_centers(data_path, whisker_spot_model_path, force_cpu):
    model = load_model(whisker_spot_model_path, force_cpu)

    inc_whiskers = []
    for image in tqdm(data_path.iterdir()):
        predictions = whisker_detector_predict(Image.open(image).convert("RGB"), model, force_cpu)
        if predictions is not None:
            inc_whiskers.append((image.name, predictions))

    return inc_whiskers


def hausdorff_distance(X_2d, Y_2d):
    X = np.zeros((X_2d.shape[0], 3))
    X[:, :-1] = X_2d

    Y = np.zeros((Y_2d.shape[0], 3))
    Y[:, :-1] = Y_2d

    reg = deformable_registration(**{"X": X, "Y": Y})
    result = reg.register()
    return max(pcu.hausdorff(X, result[0]), pcu.hausdorff(result[0], X))


def sinkhorn_distance(X_2d, Y_2d):
    X = np.zeros((X_2d.shape[0], 3))
    X[:, :-1] = X_2d

    Y = np.zeros((Y_2d.shape[0], 3))
    Y[:, :-1] = Y_2d

    M = pcu.pairwise_distances(X, Y)

    w_X = (X[:, 1] - 1) * -1
    w_Y = (Y[:, 1] - 1) * -1

    P = pcu.sinkhorn(w_X, w_Y, M, eps=1e-3)
    return (M * P).sum()


def predict(
    query_image_set_path,
    gallery_path,
    whisker_spot_model_path,
    n,
    algorithm="sinkhorn",
    lion_subset=None,
    force_cpu=False,
):
    assert algorithm == "sinkhorn" or algorithm == "cpd"

    query_image_set_path = Path(query_image_set_path)
    gallery_path = Path(gallery_path)
    whisker_spot_model_path = Path(whisker_spot_model_path)

    (right_whiskers, right_labels) = torch.load(gallery_path / "right_data.pt")
    (left_whiskers, left_labels) = torch.load(gallery_path / "left_data.pt")

    inc_whiskers = get_whisker_centers(query_image_set_path, whisker_spot_model_path, force_cpu)

    predictions = {}
    for (img_name, probe_whisker) in tqdm(inc_whiskers):
        distances = []

        for gal_whiskers, gal_labels in [
            (right_whiskers, right_labels),
            (left_whiskers, left_labels),
        ]:
            for gal_whisker, gal_label in zip(gal_whiskers, gal_labels):
                if lion_subset is None or gal_label in lion_subset:
                    if algorithm == "cpd":
                        cloud_dist = hausdorff_distance(gal_whisker, probe_whisker)
                    else:
                        cloud_dist = sinkhorn_distance(gal_whisker, probe_whisker)

                    cloud_dist = 1 / cloud_dist if cloud_dist > 0 else 0
                    distances.append((cloud_dist, gal_label))

        distances = sorted(distances, key=lambda x: x[0])
        distances = list(zip(*distances))
        predictions[img_name] = get_k_neighbors(distances[1], distances[0], n)

    return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC Face Prediction")
    parser.add_argument(
        "query_image_set_path", help="Path to the folder containing the labeled images"
    )
    parser.add_argument("n", default=3, help="How many lions to retrive per image")
    parser.add_argument(
        "gallery_path",
        help="Path to the gallery, contains: right_data.pt, left_data.pt and whisker_image_ids.pt",
    )
    parser.add_argument(
        "whisker_spot_model_path",
        help="Path to the folder where the models and gallery will be saved",
    )
    parser.add_argument(
        "--algorithm",
        default="sinkhorn",
        help="Which algorithm to use for matching: sinkhorn or cpd",
    )
    parser.add_argument(
        "--lion_subset",
        default=None,
        help="Comma separated list of the lion ids to be matched agianst. Default: All lions",
    )

    parser.add_argument("--cpu", dest="cpu", help="Force model to use CPU", action="store_true")

    args = parser.parse_args()

    lion_subset = (
        [int(lion_id) for lion_id in args.lion_subset.split(",")] if args.lion_subset else None
    )

    tic = time.time()
    results = predict(
        args.query_image_set_path,
        args.gallery_path,
        args.whisker_spot_model_path,
        float(args.n),
        args.algorithm,
        lion_subset,
        args.cpu,
    )
    toc = time.time()
    print(results, f"Done in {toc - tic:.2f} seconds!")
