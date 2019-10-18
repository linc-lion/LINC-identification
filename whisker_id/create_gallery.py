import os
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from utils import get_image_ids, load_model, whisker_detector_predict


@torch.no_grad()
def process_gallery(data_path, force_cpu, whisker_spot_model_path):

    model = load_model(whisker_spot_model_path, force_cpu)

    right_whiskers = []
    right_labels = []
    left_whiskers = []
    left_labels = []
    for lion in tqdm(data_path.iterdir()):
        if lion.name == ".DS_Store":
            os.system("rm {}".format(lion))
            continue
        for whisker_folder in lion.iterdir():
            if whisker_folder.name == ".DS_Store":
                os.system("rm {}".format(whisker_folder))
                continue
            for image in whisker_folder.iterdir():
                predictions = whisker_detector_predict(
                    Image.open(image).convert("RGB"), model, force_cpu
                )
                if predictions is not None and predictions.shape[0] > 1:
                    if "left" in whisker_folder.name:
                        left_whiskers.append(predictions)
                        left_labels.append(int(lion.name))
                    elif "right" in whisker_folder.name:
                        right_whiskers.append(predictions)
                        right_labels.append(int(lion.name))

    return right_whiskers, right_labels, left_whiskers, left_labels


def create_gallery(
    data_path, output_path, whisker_spot_model_path, current_gallery_path=None, force_cpu=False
):
    data_path = Path(data_path)
    output_path = Path(output_path)
    gallery_path = output_path / "gallery-{}".format(datetime.today().strftime("%m-%d-%y"))
    gallery_path.mkdir(exist_ok=True)

    right_whiskers, right_labels, left_whiskers, left_labels = process_gallery(
        data_path, force_cpu, whisker_spot_model_path
    )

    if current_gallery_path:
        image_ids = torch.load(current_gallery_path / "image_ids.pt")
        image_ids = get_image_ids(data_path, image_ids)
        (current_right_whiskers, current_right_labels) = torch.load(
            current_gallery_path / "right_data.pt"
        )
        (current_left_whiskers, current_left_labels) = torch.load(
            current_gallery_path / "left_data.pt"
        )

        right_whiskers.append(current_right_whiskers)
        right_labels.append(current_right_labels)

        left_whiskers.append(current_left_whiskers)
        left_labels.append(current_left_labels)
    else:
        image_ids = get_image_ids(data_path)

    torch.save(image_ids, gallery_path / "image_ids.pt")
    torch.save((left_whiskers, left_labels), gallery_path / "left_data.pt")
    torch.save((right_whiskers, right_labels), gallery_path / "right_data.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC whisker gallery creation")
    parser.add_argument("data_path", help="Path to the folder containing the labeled images")
    parser.add_argument("output_path", help="Path to the folder where gallery will be saved")
    parser.add_argument(
        "whisker_spot_model_path",
        help="Path to the folder where the models and gallery will be saved",
    )
    parser.add_argument(
        "--current_gallery_path",
        default=None,
        help="The gallery to be updated by adding the lions provided on the data path",
    )
    parser.add_argument("--cpu", dest="cpu", help="Force model to use CPU", action="store_true")
    args = parser.parse_args()

    tic = time.time()
    create_gallery(args.data_path, args.output_path, args.whisker_spot_model_path, args.cpu)
    toc = time.time()
    print(f"Done in {toc - tic:.2f} seconds!")
