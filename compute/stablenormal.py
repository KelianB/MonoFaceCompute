from glob import glob
import re
from pathlib import Path

import argparse
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np 


if __name__ == "__main__":
    """
        Use StableNormal for monocular normal predictions.
        https://github.com/Stable-X/StableNormal
    """
    parser = argparse.ArgumentParser(description="Compute normals with StableNormal.")
    parser.add_argument("--images", type=str, help="Pattern to images (glob)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    image_paths = glob(args.images)
    image_paths.sort(key=lambda f: int(re.sub("\D", "", f)))

    output_dir = Path(args.output_dir)

    # Create predictor instance
    predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
    if torch.cuda.is_available():
        predictor.to(torch.device("cuda"))

    for path in tqdm(image_paths):
        img = Image.open(path)

        # Apply the model to the image
        output = predictor(img, mode="stable")

        output = np.array(output) # (H,W,3)
        output[..., 0] = 255 - output[..., 0] # flip x

        # Save the result
        output = Image.fromarray(output, mode="RGB")
        output.save(output_dir / (Path(path).stem + ".png"))
