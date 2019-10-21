import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from dataset_utils import preprocess_image
from linc_detection.models import detection

# Params

filters = {
    0: {3: "cv-f"},
    1: {1: "cv-dl", 2: "cv-dr", 3: "cv-f"},
    2: {1: "cv-dl", 2: "cv-dr", 3: "cv-f", 4: "cv-sl", 5: "cv-sr"},
}


def dataset_generator(
    input_path, output_path, cv_model_path, force_cpu=False, transform=2, filter_type=1
):
    """Creates a copy of a dataset, applies filters and transformations for each image.
    For info on the dataset folder structure look at README.
    Parameters
    ----------
    input_path : str
        Path to the original dataset
    output_path : str
        Path to the location where the copy will be created
    cv_model_path : str
        Path to the body parts model checkpoint (.pth)
    force_cpu : bool, optional
        Force to use CPU
    transform : {0, 1, 2}, optional, default: 2
        Select which transformations to apply:
             0-None
             1-Align eyes
             2-Align eyes + zoom
    filter_type : {0, 1, 2}, optional, default: 1
        Select which filter to apply:
            0-only frontal images
            1-frontal & diagonal images
            2-all images
    """

    filter_labels = filters[filter_type]

    # Load Model

    device = "cuda" if torch.has_cuda and not force_cpu else "cpu"
    print(f"Running inference on {device} device")

    print("Loading checkpoint from hardrive... ", end="", flush=True)
    checkpoint = torch.load(cv_model_path, map_location=device)
    label_names = checkpoint["label_names"]
    print("Done.")

    print("Building model and loading checkpoint into it... ", end="", flush=True)
    model = detection.fasterrcnn_resnet50_fpn(
        num_classes=len(label_names) + 1, pretrained_backbone=False
    )
    model.to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()
    print("Done.")

    # Create dataset

    deleted_images = []

    print("Creating dataset...")
    output_path = Path(output_path) / "dataset-{}".format(datetime.today().strftime("%m-%d-%y"))
    output_path.mkdir()

    input_path = Path(input_path)

    for image_path in tqdm(input_path.rglob("*.jpg")):
        image = Image.open(image_path)
        image = preprocess_image(model, filter_labels.keys(), device, transform, image)
        if image is not None:
            output_image_path = output_path.joinpath(image_path.parent.name, image_path.name)
            if not output_image_path.parent.exists():
                output_image_path.parent.mkdir(parents=True)
            image.save(output_image_path)
        else:
            deleted_images.append(image_path)

    print("Done! {} were deleted. \n{}".format(len(deleted_images), deleted_images))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC Dataset Generator")
    parser.add_argument("input_path", help="Path to the folder containing the labeled images")
    parser.add_argument(
        "output_path", default="./new_dataset/", help="Path to destination of the new dataset"
    )
    parser.add_argument(
        "model_checkpoint_path", help="Path to the body parts object detector checkpoint"
    )
    parser.add_argument("--cpu", dest="cpu", help="Force model to use CPU", action="store_true")
    parser.add_argument(
        "--transform",
        default=2,
        type=int,
        help="Select alignment and crop: 0-None, 1-Align eyes, 2-Align eyes + zoom",
    )
    parser.add_argument(
        "--filter",
        default=1,
        type=int,
        help="Select which filter to apply: 0-only frontal, 1-frontal & diagonal, 2-all images",
    )
    args = parser.parse_args()
    tic = time.time()
    dataset_generator(
        args.input_path,
        args.output_path,
        args.model_checkpoint_path,
        args.cpu,
        int(args.transform),
        int(args.filter),
    )
    toc = time.time()
    print(f"Done in {toc - tic:.2f} seconds!")
