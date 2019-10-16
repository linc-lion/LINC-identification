import re
from collections import defaultdict

import numpy as np
import torch
from torchvision import transforms

to_tensor = transforms.ToTensor()


@torch.no_grad()
def whisker_detector_predict(pil_image, model, device, draw_confidence_threshold=0.65):

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

        lion_images = image_ids[lion_id]
        if image_id not in lion_images:
            lion_images.append(image_id)
    return image_ids
