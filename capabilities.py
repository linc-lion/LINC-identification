from pathlib import Path

import torch


def capabilities(gallery_path, algorithm):
    """Displays all the lions and images included in the database.
    Parameters
    ----------
    gallery_path : str
        Path to the gallery. Contains: embeddings.pt, face_image_ids.pt and labels.pt
    algorithm : {"face", "whisker"}
    Returns
    ----------
    dict
        Key: Lion ID
        Value: Array containing the image ids.
    """
    assert algorithm == "face" or algorithm == "whisker"
    gallery_path = Path(gallery_path)
    return torch.load(gallery_path / f"{algorithm}_image_ids.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC capabilities")
    parser.add_argument("gallery_path", help="Path to the gallery")
    parser.add_argument(
        "algorithm", help="Algorithm to return the capaiblities from (face or whisker)."
    )
    args = parser.parse_args()
    print(capabilities(args.gallery_path, args.algorithm))
