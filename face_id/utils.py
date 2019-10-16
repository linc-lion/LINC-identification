import re
from collections import defaultdict

from fastai.callbacks.hooks import hook_output
from fastai.widgets import DatasetFormatter


def get_embeddings(learn, fix_dl, **kwargs):
    hook = hook_output(list(learn.model.modules())[-3])
    actns = DatasetFormatter.get_actns(learn, hook=hook, dl=fix_dl, **kwargs)

    return actns


def get_image_ids(input_path, image_ids=defaultdict(list)):
    for image_path in input_path.rglob("*.jpg"):
        lion_images = image_ids[int(image_path.parent.name)]
        image_id = int(re.search(r"image_(\d*)", image_path.name).group(1))
        if image_id not in lion_images:
            lion_images.append(image_id)
    return image_ids
