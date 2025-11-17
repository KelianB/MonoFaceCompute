import os
import time
import logging
from pathlib import Path
from typing import Dict, List
import argparse
import yaml
from typing import List

from compute.utils import DotDict, setup_logging, run_cmd, run_cmds

DEFAULT_FOCAL_LENGTH = 1200
MAX_THREADS = 0 # set to 0 to disable multithreading

PATH_ROOT = Path("./").absolute()
PATH_MODNET = PATH_ROOT / "submodules/MODNet"
PATH_FACE_PARSING = PATH_ROOT / "submodules/face-parsing.PyTorch"
PATH_INFERNO = PATH_ROOT / "submodules/INFERNO"
PATH_SMIRK = PATH_ROOT / "submodules/SMIRK"
PATH_OMNIDATA = PATH_ROOT / "submodules/omnidata"
PATH_DSINE = PATH_ROOT / "submodules/DSINE/projects/dsine"
PATH_SAPIENS = PATH_ROOT / "submodules/sapiens/lite/demo"

def title(s):
    return "\n" + "#"*50 + f"\n{s}\n" + "#"*50

def run(sequences: Dict, shape_sequence: str, output_dir: str, resize: int,
               crop_scale: float, crop_mode: str, smooth_tracking: bool,
               tracker: str, shape_tracker: str, steps: List[str],
               normals_estimator: str):

    setup_logging()

    if MAX_THREADS > 0:
        num_threads = min(len(sequences), 4) # os.cpu_count()
    else:
        num_threads = 0

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    seq_dirs = {}
    frame_patterns: Dict[str, Path] = {}
    crop_patterns: Dict[str, Path] = {}
    mask_patterns: Dict[str, Path] = {}
    deca_dirs: Dict[str, Path] = {}
    segmentation_dirs: Dict[str, Path] = {}
    normals_dirs: Dict[str, Path] = {}
    for seq_name in sequences:
        seq_dir = output_dir / str(seq_name)
        seq_dirs[seq_name] = seq_dir
        frame_patterns[seq_name] = seq_dir / "image" / "%d.png"
        crop_patterns[seq_name] = seq_dir / "crops" / "%d.png"
        mask_patterns[seq_name] = seq_dir / "mask" / "%d.png"
        deca_dirs[seq_name] = seq_dir / "deca"
        segmentation_dirs[seq_name] = seq_dir / "semantic"
        normals_dirs[seq_name] = seq_dir / f"normals_{normals_estimator}"

    start_time = time.time()

    ############################################################
    if "extract" in steps:
        logging.info(title("Extract, crop and resize frames"))
        cmds = []
        for seq_name, seq in sequences.items():
            source = seq["source"]
            frame_patterns[seq_name].parent.mkdir(exist_ok=True, parents=True)
            # mpdecimate and setpts remove duplicate frames
            cmd = f"ffmpeg -y {'-pattern_type glob' if '*' in source else ''} -i '{source}'" +\
                f" -vf 'mpdecimate, setpts=N/FRAME_RATE/TB'" +\
                f" '{frame_patterns[seq_name]}'"
            cmds.append(cmd)
        run_cmds(cmds, num_threads)

    ############################################################
    if "crop" in steps:
        logging.info(title("Detect & crop faces"))
        cmds = []
        for seq_name, seq in sequences.items():
            segmentation_dirs[seq_name].mkdir(exist_ok=True, parents=True)
            cmd = f"python compute/detect_crops.py --batch_size 8 --resize {resize} --scale {crop_scale} " +\
                f"--input '{frame_patterns[seq_name].parent}' --output '{crop_patterns[seq_name].parent}'"
            cmd += f" --mode {seq.crop_mode if 'crop_mode' in seq else crop_mode}" +\
                  (f" --fixed_box {' '.join(map(str, seq.fixed_box))}" if "fixed_box" in seq else "") +\
                   f" --face_selection_strategy {seq.face_selection_strategy}"
            
            cmds.append(cmd)
        run_cmds(cmds, cwd=PATH_ROOT)

    ############################################################
    if "matte" in steps:
        logging.info(title("Matting (background/foreground segmentation)"))
        cmds = []
        for seq_name, seq in sequences.items():
            mask_patterns[seq_name].parent.mkdir(exist_ok=True, parents=True)
            cmd = f"python -m demo.image_matting.colab.inference --input-path '{crop_patterns[seq_name].parent}'" +\
                f" --output-path '{mask_patterns[seq_name].parent}' --ckpt-path ./pretrained/modnet_webcam_portrait_matting.ckpt"
            cmds.append(cmd)
        run_cmds(cmds, num_threads, cwd=PATH_MODNET)

    ############################################################
    if "segment" in steps:
        logging.info(title("Semantic face segmentation"))
        cmds = []
        for seq_name, seq in sequences.items():
            segmentation_dirs[seq_name].mkdir(exist_ok=True, parents=True)
            cmd = f"python test.py --input_dir '{crop_patterns[seq_name].parent}' --output_dir '{segmentation_dirs[seq_name]}'"
            cmds.append(cmd)
        run_cmds(cmds, cwd=PATH_FACE_PARSING)

    ############################################################
    if "normals" in steps:
        assert normals_estimator in ["omnidata", "dsine", "stablenormal", "sapiens"]
        
        logging.info(title("Normals estimation"))
        cmds = []
        for seq_name, seq in sequences.items():
            normals_dirs[seq_name].mkdir(exist_ok=True, parents=True)
    
            if normals_estimator == "omnidata":
                cmd = f"python estimate_normals.py --task normal --img_path '{crop_patterns[seq_name].parent}' --output_path '{normals_dirs[seq_name]}'"
            elif normals_estimator == "dsine":
                cmd = f"python test_minimal_custom.py ./experiments/exp001_cvpr2024/dsine.txt --input_dir '{crop_patterns[seq_name].parent}' --output_dir '{normals_dirs[seq_name]}'"
            elif normals_estimator == "stablenormal":
                cmd = f"conda run -n StableNormal --no-capture-output python compute/stablenormal.py --images '{crop_patterns[seq_name].parent / '*'}' --output_dir '{normals_dirs[seq_name]}'"
            elif normals_estimator == "sapiens":
                checkpoint = "../../sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"
                cmd = f"conda run -n sapiens_lite --no-capture-output python vis_normal.py {checkpoint} --input '{crop_patterns[seq_name].parent}' --output_root '{normals_dirs[seq_name]}' --batch_size {8} --subsample {normals_subsample}"

            cmds.append(cmd)
        run_cmds(cmds, cwd={"omnidata": PATH_OMNIDATA, "dsine": PATH_DSINE, "stablenormal": PATH_ROOT, "sapiens": PATH_SAPIENS}[normals_estimator])

    ############################################################
    if "landmarks" in steps:
        logging.info(title("Landmarks detection"))
        cmds = []
        for seq_name, seq in sequences.items(): 
            cmd = f"python compute/detect_landmarks.py --images '{str(crop_patterns[seq_name]).replace('%d', '*')}' --output_dir '{seq_dirs[seq_name]}'"
            cmds.append(cmd)
        run_cmds(cmds, num_threads, cwd=PATH_ROOT)

    # All the remaining steps run on GPU, so we don't do multi-threading for those

    ############################################################
    if "track" in steps:
        logging.info(title(f"FLAME parameter estimation using tracker `{tracker}`"))

        def track(tracker: str, seq_name: str, output_name: str):
            if tracker == "DECA":
                cmd = f"python inferno_apps/EMOCA/demos/test_emoca_on_images.py --input_folder '{crop_patterns[seq_name].parent}'" +\
                    f" --output_folder '{seq_dirs[seq_name]}' --save_codes True --save_images False --model_name DECA --batch_size 16 --output_name {output_name}"
            elif tracker == "EMOCA":
                cmd = f"python inferno_apps/EMOCA/demos/test_emoca_on_images.py --input_folder '{crop_patterns[seq_name].parent}'" +\
                    f" --output_folder '{seq_dirs[seq_name]}' --save_codes True --save_images False --model_name EMOCA_v2_lr_mse_20 --batch_size 16 --output_name {output_name}"
            elif tracker == "FaceReconstruction":
                cmd = f"python inferno_apps/FaceReconstruction/demo/demo_face_rec_on_images.py --input_folder '{crop_patterns[seq_name].parent}'" +\
                    f" --output_folder '{seq_dirs[seq_name]}' --save_codes True --save_images False --batch_size 16 --output_name {output_name}"
            elif tracker == "SMIRK":
                cmd = f"python demo_images.py --input_folder '{crop_patterns[seq_name].parent}' --output_folder '{seq_dirs[seq_name]}' --batch_size 16 --output_name {output_name}"
                cmd = f"PYTHONPATH={PATH_SMIRK} {cmd}"
            else:
                raise ValueError(f"Unknown tracker '{tracker}'")
            return cmd

        cmds = [track(tracker, seq_name, "flame_params_inferred.json") for seq_name in sequences.keys()]
        run_cmds(cmds, cwd=PATH_SMIRK if tracker == "SMIRK" else PATH_INFERNO)

        # If shape_tracker is different, track the shape sequence again using that tracker 
        if shape_tracker != tracker:
            run_cmd(track(shape_tracker, shape_sequence, "flame_params_inferred_shape.json"), cwd=PATH_SMIRK if shape_tracker == "SMIRK" else PATH_INFERNO)


    ############################################################
    if "optimize" in steps:
        cx = resize / 2
        cy = resize / 2

        # Run optimization with 0 iters just to convert the inferred shape from shape_tracker to the same format
        if shape_tracker != tracker:
            cmd = f"python compute/optimize.py --path '{seq_dirs[shape_sequence]}' --size {resize} --iterations 0" +\
                f" --input_name flame_params_inferred_shape.json --save_name flame_params_optimized_shape"
            run_cmd(cmd, cwd=PATH_ROOT)

        logging.info(title(f"Fit FLAME parameter for one video: {shape_sequence}"))    
        fl = sequences[shape_sequence].get("fl", DEFAULT_FOCAL_LENGTH)
        cmd = f"python compute/optimize.py --path '{seq_dirs[shape_sequence]}' --input_name flame_params_inferred.json" +\
            f" --cx {cx} --cy {cy} --fx {fl} --fy {fl} --size {resize} {'--smooth' if smooth_tracking else ''} --optimize_shape"
        if shape_tracker != tracker:
            cmd += f" --shape_from '{seq_dirs[shape_sequence] / 'flame_params_optimized_shape.json'}'"
        run_cmd(cmd, cwd=PATH_ROOT)

        logging.info(title("Fit FLAME parameter for other videos, while keeping shape parameter fixed"))
        cmds = []
        for seq_name, seq in sequences.items():
            if seq_name == shape_sequence:
                continue
            fl = seq.get("fl", DEFAULT_FOCAL_LENGTH)
            cmd = f"python compute/optimize.py --path '{seq_dirs[seq_name]}'" +\
                f" --shape_from '{seq_dirs[shape_sequence] / 'flame_params_optimized.json'}'" +\
                f" --cx {cx} --cy {cy} --fx {fl} --fy {fl} --size {resize} {'--smooth' if smooth_tracking else ''}"
            cmds.append(cmd)
        run_cmds(cmds, cwd=PATH_ROOT)

    end_time = time.time()
    logging.info(f"Done! Time taken: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")

    if "visualize" in steps:
        logging.info(title("Visualize tracking"))
        cmds = []
        for seq_name, seq in sequences.items():
            cmd = f"python compute/visualize.py --path '{seq_dirs[seq_name]}'" +\
                f" --tracking_name flame_params_optimized.json" +\
                f" --output_dir '{seq_dirs[seq_name]}/visualize'"
            cmds.append(cmd)
        run_cmds(cmds, cwd=PATH_ROOT)

    end_time = time.time()
    logging.info(f"Done! Time taken: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="YAML configuration file.")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise ValueError(f"Provided dataset '{args.dataset}' not found.")
    
    with open(args.dataset, "r") as file:
        cfg = yaml.safe_load(file)
    cfg = DotDict(cfg)

    # Set defaults
    if "steps" not in cfg:
        cfg.steps = ["extract", "crop", "matte", "segment", "landmarks", "track", "optimize"]
    if "tracker" not in cfg:
        cfg.tracker = "EMOCA"
    if "shape_tracker" not in cfg:
        cfg.shape_tracker = cfg.tracker
    if "smooth_tracking" not in cfg:
        cfg.smooth_tracking = True
    for seq in cfg.sequences.values():
        if "face_selection_strategy" not in seq:
            seq.face_selection_strategy = "max_confidence"

    if "base_dir" in cfg:
        for seq in cfg.sequences.values():
            seq.source = os.path.join(cfg.base_dir, seq.source)

    if "normals_estimator" not in cfg:
        cfg.normals_estimator = "dsine"

    run(cfg.sequences, cfg.shape_sequence, cfg.output_dir, cfg.resize, cfg.crop_scale, cfg.crop_mode, cfg.smooth_tracking, cfg.tracker, cfg.shape_tracker, cfg.steps, cfg.normals_estimator)
