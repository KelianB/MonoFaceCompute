import os
import json
from pathlib import Path
from tqdm import tqdm

import argparse
import torch
import numpy as np
from imageio import imread, imsave

import inferno

from optimize import projection, load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmarks-based facial pose, expression and shape optimization")
    parser.add_argument("--path", type=str, help="Path to images and deca and landmark jsons")
    parser.add_argument("--tracking_name", type=str, default="flame_params_optimized.json", help="Name for input json")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for visualizations")

    args = parser.parse_args()

    image_path = os.path.join(args.path, "crops")  
    with open(os.path.join(args.path, args.tracking_name), 'r') as f:
        tracking_params = json.load(f)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(0)

    fps = 24
    size = 512

    # Load any version of DECA, this is just to have access to the rendering
    model_path = Path(inferno.__file__).parents[1] / "assets/EMOCA/models/DECA"
    flame, render = load_model(model_path, device, render_size=size)

    shape_params = torch.tensor(tracking_params["shape_params"], dtype=torch.float, device=device).unsqueeze(0)
    cam_intrinsics = torch.tensor(tracking_params["intrinsics"], dtype=torch.float, device=device)
    cam_intrinsics *= 2
    cam_intrinsics[0] *= -1
    cam_intrinsics[2] -= 1
    cam_intrinsics[3] -= 1

    for idx, item in tqdm(enumerate(tracking_params["frames"]), total=len(tracking_params["frames"])):
        name = str(idx+1)
        image = imread(f"{image_path}/{name}.png") # (H, W, 3)
        image = torch.tensor(image / 255, dtype=torch.float)
        # resize image
        if image.shape[0] != size or image.shape[1] != size:
            image = torch.nn.functional.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)

        pose_params = torch.tensor(item["pose"], dtype=torch.float, device=device).unsqueeze(0)
        expr_params = torch.tensor(item["expression"], dtype=torch.float, device=device).unsqueeze(0)
        world_mat = torch.tensor(item["world_mat"], dtype=torch.float, device=device).unsqueeze(0)
        per_frame_translation = pose_params.shape[-1] == 18
        flame_pose = pose_params[:, :-3] if per_frame_translation else pose_params
        verts_p, landmarks2d_p, landmarks3d_p, landmarks2d_mediapipe = flame(shape_params=shape_params, expression_params=expr_params, pose_params=flame_pose[:,:9], eye_pose_params=flame_pose[:,9:])

        # Add per-frame translation
        if per_frame_translation:
            pose_translation = pose_params[:, -3:].unsqueeze(1)
            verts_p += pose_translation
            landmarks3d_p += pose_translation
            landmarks2d_p += pose_translation
            landmarks2d_mediapipe += pose_translation

        trans_verts = projection(verts_p, cam_intrinsics, world_mat)

        shape_render = render.render_shape(verts_p, trans_verts).squeeze(0)       
        shape_render = shape_render.permute(1, 2, 0).cpu() # (3, H, W) to (H, W, 3)

        visualize_image = torch.cat((image, shape_render), dim=1)
        visualize_image = (visualize_image * 255).numpy().astype(np.uint8)
        imsave(str(output_dir / f"{idx+1:05d}.png"), visualize_image)

    print("Creating video")
    os.system(f"/usr/bin/ffmpeg -y -framerate {fps} -pattern_type glob -i '{output_dir / '*.png'}' -c:v libx264 -pix_fmt yuv420p {output_dir / 'video.mp4'}")
    print("Done")
