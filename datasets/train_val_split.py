import random
import shutil
from pathlib import Path


def train_val_split(input_path, min_images, train_size):
    """Splits a dataset into train, validation and not used.
    Parameters
    ----------
    input_path : str
        Path to the original dataset
    min_images : int
        Minimum amount of images required for a lion to be included on the dataset.
    train_size : float
        Proportion of the dataset to include in the train split (between 0 and 1).
    """

    input_path = Path(input_path)

    train_root = input_path.joinpath("train")
    train_root.mkdir()
    val_root = input_path.joinpath("valid")
    val_root.mkdir()
    not_used_root = input_path.joinpath("not_used")
    not_used_root.mkdir()

    for lion_dir in input_path.iterdir():
        if lion_dir.name not in ["train", "valid", "not_used"]:
            lion_imgs = list(lion_dir.iterdir())
            if len(lion_imgs) <= min_images:
                shutil.move(str(lion_dir), input_path.joinpath("not_used"))
            else:
                random.shuffle(lion_imgs)
                idx_limit = int(train_size * len(lion_imgs))
                train_images = lion_imgs[:idx_limit]
                val_images = lion_imgs[idx_limit:]

                for root_dir, img_set in [(train_root, train_images), (val_root, val_images)]:
                    for img in img_set:
                        lion_new_dir = root_dir.joinpath(lion_dir.name)
                        if not lion_new_dir.exists():
                            lion_new_dir.mkdir()
                        shutil.move(str(img), lion_new_dir)

                lion_dir.rmdir()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LINC Validation splitter (inplace)")
    parser.add_argument("input_path", help="Path to the folder containing the labeled images")
    parser.add_argument(
        "--min_images",
        default=3,
        help="The minimum amount of images required for a lion to be used.",
        type=int,
    )
    parser.add_argument(
        "--train_size",
        default=0.8,
        type=float,
        help="Proportion of the dataset to include in the train split (between 0 and 1).",
    )
    args = parser.parse_args()

    train_val_split(args.input_path, int(args.min_images), float(args.train_size))
