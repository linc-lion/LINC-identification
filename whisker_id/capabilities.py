from pathlib import Path

import torch


def capabilities(gallery_path):
    gallery_path = Path(gallery_path)
    return torch.load(gallery_path / "whisker_image_ids.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC capabilities")
    parser.add_argument(
        "gallery_path",
        help="Path to the gallery. Contains: left_data.pt, whisker_image_ids.pt and right_data.pt",
    )
    args = parser.parse_args()
    print(capabilities(args.gallery_path))
