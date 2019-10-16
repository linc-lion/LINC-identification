from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torchvision
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from linc_detection.models import detection
from utils import preprocess_image

convert_to_pil = torchvision.transforms.ToPILImage()

# Params

filters = {
    0: {3: "cv-f"},
    1: {1: "cv-dl", 2: "cv-dr", 3: "cv-f"},
    2: {1: "cv-dl", 2: "cv-dr", 3: "cv-f", 4: "cv-sl", 5: "cv-sr"},
}


def dataset_generator(
    input_path, output_path, cv_model_path, force_cpu=False, transform=2, filter_type=1
):

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
    preprocess_image_args = partial(
        preprocess_image, model, filter_labels.keys(), device, transform
    )

    image_ds = ImageFolder(input_path, preprocess_image_args)
    deleted_images = []

    print("Creating dataset...")
    output_path = Path(output_path) / "dataset-{}".format(datetime.today().strftime("%m-%d-%y"))
    output_path.mkdir()

    for idx, (image, label) in enumerate(tqdm(image_ds)):
        if image is not None:
            original_path = Path(image_ds.samples[idx][0])
            image_path = output_path.joinpath(original_path.parent.name, original_path.name)
            if not image_path.parent.exists():
                image_path.parent.mkdir(parents=True)
            convert_to_pil(image).save(image_path)
        else:
            deleted_images.append(image_ds.samples[idx][0])

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

    dataset_generator(
        args.input_path,
        args.output_path,
        args.model_checkpoint_path,
        args.cpu,
        int(args.transform),
        int(args.filter),
    )
