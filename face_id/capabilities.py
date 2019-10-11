from pathlib import Path

import torch

from config import configuration


def capabilities():
    gallery_path = Path(configuration["GALLERY_PATH"])
    return torch.load(gallery_path / "image_ids.pt")


if __name__ == "__main__":

    print(capabilities())
