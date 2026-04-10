import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
import cv2
import numpy as np
import imageio

import face_alignment
from inferno.datasets.ImageTestDataset import TestDataCustom

from utils import gaussian_kernel, apply_featurewise_conv1d

CROP_MODES = ["none", "smooth", "constant", "fixed"]
"""
- none: keep the whole image
- smooth: detect faces and smooth the detections
- constant: detect faces and automatically compute a constant crop using the extrema of all detections within the video
- fixed: manually pass a constant crop box
"""

FACE_SELECTION_STRATEGY = ["max_confidence", "leftmost", "rightmost"]
# TODO implement other strategies: largest, longest, closest to previous frame, ...

CROP_MODE_CONSTANT_Q = 0.05 # quantile for mode=constant
CONFIDENCE_THRESHOLD = 0.6 # min threshold for resolving multiple detections on the same frame

@torch.no_grad()
def detect_faces(dataloader, face_selection_strategy):
    raw_boxes = []
    num_detections_stats = dict()
    image_idx = 0
    # Record frames to skip before of missing detection
    valid_ids = []

    for batch in tqdm(dataloader):
        imgs = batch["image"]

        # Face detection
        boxes_batch = fa.face_detector.detect_from_batch(imgs.permute(0,3,1,2)) # BHWC to BCHW

        for boxes, name in zip(boxes_batch, batch["image_name"]):
            boxes = [torch.tensor(box, dtype=torch.float) for box in boxes]
            n_faces = len(boxes)

            num_detections_stats[n_faces] = num_detections_stats.get(n_faces, 0) + 1

            if n_faces == 0:
                print(f"No face detected for frame {name}")
                image_idx += 1
                continue

            # If there are multiple detections, remove those under the confidence threshold
            if n_faces > 1:
                # print(f"Detected {n_faces} faces - confidence values: " + ', '.join(f"{b[-1]:.2f}" for b in boxes))
                boxes = [box for box in boxes if box[-1] > CONFIDENCE_THRESHOLD]
                n_faces = len(boxes)

            # If there are still multiple detections, select one
            if n_faces > 1:
                if face_selection_strategy == "max_confidence":
                    bbox = sorted(boxes, key=lambda box: box[-1])[-1]
                elif face_selection_strategy == "leftmost":
                    bbox = sorted(boxes, key=lambda box: box[0])[0]
                elif face_selection_strategy == "rightmost":
                    bbox = sorted(boxes, key=lambda box: box[0])[-1]
                else:
                    raise ValueError(f"Unknown face selection strategy '{face_selection_strategy}'")
            else:
                bbox = boxes[0]
            raw_boxes.append(bbox[:4])
            valid_ids.append(image_idx)
            image_idx += 1
    
    for n_faces in sorted(num_detections_stats.keys()):
        print(f"Detected {n_faces} faces on {num_detections_stats[n_faces]} images.")

    return torch.stack(raw_boxes), valid_ids

@torch.no_grad()
def boxes_to_crops(raw_boxes, image_size, scale=1.0):
    crops = []
    H, W = image_size
    for bbox in raw_boxes:
        x1, y1, x2, y2 = bbox
        # Make the box a square
        s = (x2 - x1 + y2 - y1) / 2
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # Scale it up
        s *= scale
        x, y = center_x - s/2, center_y - s/2
        # Ensure the crop stays within the image
        s = s.clamp(min=0, max=min(H, W))
        x = x.clamp(min=0, max=W-s)
        y = y.clamp(min=0, max=H-s)
        # Round
        crops.append(torch.stack((x,y,s), dim=-1).round().int())

    return torch.stack(crops)

def export_crops(dataloader, crop_boxes, output_dir: Path, resize: int, debug_dir=None):
    image_idx = 0
    for batch in tqdm(dataloader):
        # for img, name in zip(batch["image"], batch["image_name"]):
        for img in batch["image"]:
            H, W, _ = img.shape
            x, y, s = crop_boxes[image_idx].tolist()

            if debug_dir:
                img_debug = img.numpy().astype(np.uint8)
                cv2.rectangle(img_debug, (x, y), (x+s, y+s), (0, 255, 0), 5)
                img_debug = cv2.resize(img_debug, (W//4, H//4))
                imageio.imsave(debug_dir / f"{image_idx+1}.png", img_debug)

            crop = img[y:y+s, x:x+s]
            crop = cv2.resize(crop.numpy().astype(np.uint8), (resize, resize), interpolation=cv2.INTER_CUBIC)
            imageio.imsave(output_dir / f"{image_idx+1}.png", crop)

            image_idx += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True, help="Path to the input images")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--resize", type=int, required=True, help="Size to resize the final crops to")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--out_path", type=str, default="output", help="Path to save the output (will be created if not exists)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--mode", type=str, default="smooth", help=f"Cropping mode", choices=CROP_MODES)
    parser.add_argument("--fixed_box", type=int, nargs="+", help="Box to use if mode=fixed")
    parser.add_argument("--debug", action="store_true", help="Export debug images of the crops")
    parser.add_argument("--face_selection_strategy", type=str, help="What strategy to use to select a box when multiple faces are detected", choices=FACE_SELECTION_STRATEGY)

    args = parser.parse_args()

    args.output = Path(args.output)
    scale = args.scale
    mode = args.mode
    assert mode in CROP_MODES

    args.output.mkdir(parents=True, exist_ok=True)

    if args.debug:
        debug_dir = args.output.parent / f"{args.output.name}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
    else:
        debug_dir = None

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
    
    dataset = TestDataCustom(args.input)
    dataset_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=TestDataCustom.collate, num_workers=4)

    H, W, _ = dataset[0]["image"].shape
    valid_ids = torch.arange(dataset_size, dtype=torch.int)

    # Do a first pass to detect face bounding boxes
    if mode == "none":
        raw_boxes = torch.tensor([[0, 0, W, H]] * dataset_size, dtype=torch.float)
    elif mode == "fixed":
        if not args.fixed_box:
            raise ValueError(f"Please specify --fixed_box to use with crop mode '{mode}'")
        if len(args.fixed_box) != 3:
            raise ValueError("--fixed_box must specify crop boxes in the following format: center_x center_y size")
        center_x, center_y, s = args.fixed_box
        raw_boxes = [torch.tensor([center_x-s/2, center_y-s/2, center_x+s/2, center_y+s/2], dtype=torch.float) for _ in range(dataset_size)]
    else:
        print("Running initial bbox detection")
        raw_boxes, valid_ids = detect_faces(dataloader, args.face_selection_strategy)

    if mode in ["none", "fixed"]:
        pass
    elif mode == "constant":
        # Compute a constant crop
        q = CROP_MODE_CONSTANT_Q
        raw_boxes[:, 0] = raw_boxes[:, 0].quantile(q)
        raw_boxes[:, 1] = raw_boxes[:, 1].quantile(q)
        raw_boxes[:, 2] = raw_boxes[:, 2].quantile(1-q)
        raw_boxes[:, 3] = raw_boxes[:, 3].quantile(1-q)   
    elif mode == "smooth":
        # Apply a gaussian filter to the box coordinates
        print("Smoothing boxes")
        conv_weights = gaussian_kernel(ksize=15, sigma=8)
        raw_boxes = apply_featurewise_conv1d(raw_boxes, conv_weights, pad_mode="replicate")

    print("Formatting boxes")
    # Make the boxes squares, scale them and clamp within image dimensions
    crop_boxes = boxes_to_crops(raw_boxes, (H,W), args.scale if mode != "none" else 1.0)

    # Re-create the data loader, skipping invalid frames
    subset = Subset(dataset, valid_ids)
    dataloader = DataLoader(subset, batch_size=args.batch_size, collate_fn=TestDataCustom.collate, num_workers=4)

    # Crop and export
    print("Cropping")
    export_crops(dataloader, crop_boxes, args.output, args.resize, debug_dir)

    print("Done")

