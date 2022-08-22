import numpy as np
import os
import argparse
import glob
from PIL import Image


def create_dummy_mask(path: str, target_path):
    mask_dir = os.path.dirname(os.path.dirname(path))
    file_name = os.path.basename(path)
    img = Image.open(path)
    h, w, _ = np.asarray(img).shape
    mask = Image.open(target_path)
    mask = mask.resize((w, h), Image.NEAREST)
    mask.save(
        mask_dir
        + "/masks/"
        + (file_name.replace(".png", "_mask.png")).replace(".jpg", "_mask.png"),
    )


if __name__ == "__main__":
    target_path = "./data/Trans10k/test.bak/masks/1_mask.png"
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    args = parser.parse_args()

    for path in glob.glob(args.dir + "/*"):
        create_dummy_mask(path, target_path)
