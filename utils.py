import re
from collections import defaultdict

import numpy as np
import torch
from fastai.callbacks.hooks import hook_output
from fastai.widgets import DatasetFormatter
from scipy.special import softmax
from torchvision import transforms

from linc_detection.models import detection

to_tensor = transforms.ToTensor()


def get_k_neighbors(neighbors, distances, n):
    lion_neighbors = []
    neighbor_dists = []
    for neighbor, distance in zip(neighbors, distances):
        if neighbor not in lion_neighbors:
            lion_neighbors.append(neighbor)
            neighbor_dists.append((1 / distance) if distance > 0 else 0)

        if len(lion_neighbors) >= n:
            break
    neighbor_dists = softmax(neighbor_dists)
    return dict(zip(lion_neighbors, neighbor_dists))


def get_embeddings(learn, fix_dl, **kwargs):
    hook = hook_output(list(learn.model.modules())[-3])
    actns = DatasetFormatter.get_actns(learn, hook=hook, dl=fix_dl, **kwargs)

    return actns


def get_image_ids(input_path, image_ids=defaultdict(list)):
    for image_path in input_path.rglob("*.jpg"):
        if image_path.parent.name.isdigit():
            lion_id = int(image_path.parent.name)
        elif image_path.parent.parent.name.isdigit():
            lion_id = int(image_path.parent.parent.name)
        else:
            raise Exception("Invalid folder structure, could not parse lion id from folder name")

        try:
            image_id = int(re.search(r"image_(\d*)", image_path.name).group(1))
        except ValueError:
            raise Exception("Invalid folder structure, could not parse image id from image name.")

        if image_id not in image_ids[lion_id]:
            image_ids[lion_id].append(image_id)
    return image_ids


@torch.no_grad()
def whisker_detector_predict(pil_image, model, force_cpu=False, draw_confidence_threshold=0.65):
    device = "cuda" if torch.has_cuda and not force_cpu else "cpu"

    image = to_tensor(pil_image).to(device)
    outputs = model([image])[0]  # We index 0 because we are using batch size 1

    scores = outputs["scores"]
    top_scores_filter = scores > draw_confidence_threshold
    top_boxes = outputs["boxes"][top_scores_filter]
    top_labels = outputs["labels"][top_scores_filter]
    if len(top_labels):
        whiskers = np.zeros((len(top_labels), 2))
        whiskers[:, 0] = np.array(top_boxes[:, 0].cpu() + top_boxes[:, 2].cpu()) / 2
        whiskers[:, 1] = np.array(top_boxes[:, 1].cpu() + top_boxes[:, 3].cpu()) / 2
        means = np.mean(whiskers, axis=0)
        whiskers = whiskers - means
        maxs = np.max(np.abs(whiskers), axis=0)
        whiskers[:, 0] = whiskers[:, 0] / maxs[0] if maxs[0] != 0 else whiskers[:, 0]
        whiskers[:, 1] = whiskers[:, 1] / maxs[1] if maxs[1] != 0 else whiskers[:, 1]
        return whiskers
    else:
        return


def load_model(whisker_spot_model_path, force_cpu):
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
    return model
