import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from linc_detection.models import detection
from utils import get_image_ids, whisker_detector_predict


def create_gallery(data_path, output_path, whisker_spot_model_path, force_cpu=False):
    data_path = Path(data_path)
    output_path = Path(output_path)

    device = "cuda" if torch.has_cuda and not force_cpu else "cpu"
    print(f"Running inference on {device} device")

    print("Loading checkpoint from hardrive... ", end="", flush=True)
    checkpoint = torch.load(whisker_spot_model_path, map_location=device)
    label_names = checkpoint["label_names"]
    print("Done.")

    print("Building model and loading checkpoint into it... ", end="", flush=True)
    model = detection.fasterrcnn_resnet50_fpn(
        num_classes=len(label_names) + 1, pretrained_backbone=False
    )
    model.to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

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
                    Image.open(image).convert("RGB"), model, device
                )
                if predictions is not None:
                    if "left" in whisker_folder.name:
                        left_whiskers.append(predictions)
                        left_labels.append(int(lion.name))
                    elif "right" in whisker_folder.name:
                        right_whiskers.append(predictions)
                        right_labels.append(int(lion.name))

    gallery_path = output_path / "gallery"
    gallery_path.mkdir()
    torch.save(get_image_ids(data_path), gallery_path / "image_ids.pt")
    torch.save((np.array(left_whiskers), np.array(left_labels)), gallery_path / "left_data.pt")
    torch.save((np.array(right_whiskers), np.array(right_labels)), gallery_path / "right_data.pt")

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC whisker gallery creation")
    parser.add_argument("data_path", help="Path to the folder containing the labeled images")
    parser.add_argument("output_path", help="Path to the folder where gallery will be saved")
    parser.add_argument(
        "whisker_spot_model_path",
        help="Path to the folder where the models and gallery will be saved",
    )
    parser.add_argument("--cpu", dest="cpu", help="Force model to use CPU", action="store_true")
    args = parser.parse_args()

    tic = time.time()
    create_gallery(args.data_path, args.output_path, args.whisker_spot_model_path, args.cpu)
    toc = time.time()
    print(f"Done in {toc - tic:.2f} seconds!")
