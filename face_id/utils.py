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
