"""
taken from feature_extraction branch from HistoBistro
"""
import argparse
import concurrent.futures
import math
import re
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import slideio
import torch
import yaml
from dataset import SlideDataset
from models.return_model import get_models, get_transforms
from options import Options
from PIL import Image
# import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import bgr_format, get_driver, get_scaling, save_hdf5, threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
extracts features from slides of specified dataset with given checkpoints
"""


def main(args):
    # adapt the feature directory in data_config
    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)
        slide_path = data_config[args.dataset]["slide_dir"]

    # Get slide files based on the provided path and file extension
    slides = sorted(glob(f"{slide_path}/**/*{args.file_extension}", recursive=True))

    if bool(args.exctraction_list) is not False:
        to_extract = pd.read_csv(args.exctraction_list).iloc[:, 0].tolist()
        slides = [file for file in slides if file.name in to_extract]

    # filter out slide files using RegEx
    slides = sorted([file for file in slides if re.search("-DX", str(file))])

    chunk_len = math.ceil(len(slides) / args.split[1])
    start = args.split[0] * chunk_len
    end = min(start + chunk_len, len(slides))
    slides = slides[start:end]

    # Get the driver for the slide file extension
    driver = get_driver(args.file_extension)

    # Load models for checkpoints
    model_dicts = []
    if args.model == ['owkin']:
        checkpoints = [Path(args.run)]
    elif Path(args.run).is_dir():
        checkpoints = Path(args.run).rglob("**/*teacher_checkpoint.pth")
    else:
        checkpoints = [Path(args.run)]
        args.run = str(Path(args.run).parent.parent.parent)

    for i, c in enumerate(checkpoints):
        output_dir = (
            Path(args.run)
            / "features"
            / f"{args.dataset}_{c.parent.name}_{args.patch_size}px_{args.model[i]}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal"
        )
        output_dir.mkdir(parents=True)
        model_dicts.append(
            {
                "model": get_models(args.model[i], c),
                "name": args.model[i],
                "transforms": get_transforms(args.model[i]),
                "save_path": output_dir,
            }
        )

        # save config
        arg_dict = vars(args)
        with open(output_dir / "config.yaml", "w") as f:
            for arg_name, arg_value in arg_dict.items():
                if isinstance(arg_value, list):
                    f.write(f"{arg_name}: {arg_value[i]}\n")
                else:
                    f.write(f"{arg_name}: {arg_value}\n")

    # Process slide files
    start = time.perf_counter()
    for slide_file in tqdm(slides, position=0, leave=False, desc="slides"):
        slide = slideio.Slide(str(slide_file), driver)
        slide_name = Path(slide_file).stem
        extract_features(output_dir, slide, slide_name, model_dicts, DEVICE, args)

    end = time.perf_counter()
    elapsed_time = end - start

    print("Time taken: ", elapsed_time, "seconds")


def process_row(wsi: np.array, scn: int, x: int, args: argparse.Namespace, slide_name: str):
    """
    Process a row of a whole slide image (WSI) and extract patches that meet the threshold criteria.

    Parameters:
    wsi (numpy.ndarray): The whole slide image as a 3D numpy array (height, width, color channels).
    scn (int): Scene number of the WSI.
    x (int): X coordinate of the patch in the WSI.
    args (argparse.Namespace): Parsed command-line arguments.
    slide_name (str): Name of the slide.

    Returns:
    pd.DataFrame: A DataFrame with the coordinates of the patches that meet the threshold.
    """

    patches_coords = pd.DataFrame()

    for y in range(0, wsi.shape[1], args.patch_size):
        # check if a full patch still 'fits' in y direction
        if y + args.patch_size > wsi.shape[1]:
            continue

        # extract patch
        patch = wsi[x : x + args.patch_size, y : y + args.patch_size, :]

        # threshold checks if it meets canny edge detection, white and black pixel criteria
        if threshold(patch, args):
            if args.save_patch_images:
                im = Image.fromarray(patch)
                im.save(
                    Path(args.save_path)
                    / "patches"
                    / str(args.downscaling_factor)
                    / slide_name
                    / f"{slide_name}_patch_{scn}_{x}_{y}.png"
                )

            patches_coords = pd.concat(
                [patches_coords, pd.DataFrame({"scn": [scn], "x": [x], "y": [y]})],
                ignore_index=True,
            )

    return patches_coords


def patches_to_feature(wsi: np.array, coords: pd.DataFrame, model_dicts: list[dict], device: torch.device):
    feats = {model_dict["name"]: [] for model_dict in model_dicts}

    with torch.no_grad():
        for model_dict in model_dicts:
            model = model_dict["model"]
            transform = model_dict["transforms"]
            model_name = model_dict["name"]

            dataset = SlideDataset(wsi, coords, args.patch_size, transform)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1, shuffle=False)

            for batch in dataloader:
                features = model(batch.to(device))
                feats[model_name] = feats[model_name] + (features.cpu().numpy().tolist())

    return feats


def extract_features(
    save_dir: str,
    slide: slideio.py_slideio.Slide,
    slide_name: str,
    model_dicts: list[dict],
    device: torch.device,
    args: argparse.Namespace,
):
    """
    Extract features from a slide using a given model.

    Args:
        slide (slideio.Slide): The slide object to process.
        slide_name (str): Name of the slide file.
        args (argparse.Namespace): Arguments containing various processing parameters.
        model_dict (dict): Dictionary containing the model, transforms, and model name.
        scene_list (list): List of scenes to process.
        device (torch.device): Device to perform computations on (CPU or GPU).

    Returns:
        None
    """

    feats = {model_dict["name"]: [] for model_dict in model_dicts}
    coords = pd.DataFrame({"scn": [], "x": [], "y": []}, dtype=int)

    if args.save_patch_images:
        (Path(args.save_path) / "patches" / str(args.downscaling_factor) / slide_name).mkdir(
            parents=True, exist_ok=True
        )

    orig_sizes = []
    scalings = []
    # iterate over scenes of the slides
    for scn in range(slide.num_scenes):
        scene_coords = pd.DataFrame({"scn": [], "x": [], "y": []}, dtype=int)
        scene = slide.get_scene(scn)
        orig_sizes.append(scene.size)

        try:
            scaling = get_scaling(args, scene.resolution[0])
            scalings.append(scaling)
        except Exception as e:
            print(e)
            print(f"Error determining resolution at slide ", slide_name, scn)
            scalings.append(-1)
            break
        # read the scene in the desired resolution

        wsi = scene.read_block(size=(int(scene.size[0] // scaling), int(scene.size[1] // scaling)))

        # revert the flipping
        # wsi=np.transpose(wsi, (1, 0, 2))

        # check if RGB or BGR is used and adapt
        # if bgr_format(slide.raw_metadata):
        #    wsi = wsi[..., ::-1]
        # print("Changed BGR to RGB!")

        # Define the main loop that processes all patches
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            # iterate over x (width) of scene
            for x in tqdm(
                range(0, wsi.shape[0], args.patch_size),
                position=1,
                leave=False,
                desc=slide_name + "_" + str(scn),
            ):
                # check if a full patch still 'fits' in x direction
                if x + args.patch_size > wsi.shape[0]:
                    continue
                future = executor.submit(process_row, wsi, scn, x, args, slide_name)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                patches_coords = future.result()
                if len(patches_coords) > 0:
                    scene_coords = pd.concat([scene_coords, patches_coords], ignore_index=True)
        coords = pd.concat([coords, scene_coords], ignore_index=True)

        if len(model_dicts) > 0:
            patch_feats = patches_to_feature(wsi, scene_coords, model_dicts, device)
            for key in patch_feats.keys():
                feats[key].extend(patch_feats[key])

    # Write data to HDF5
    if len(model_dicts) > 0:
        save_hdf5(save_dir, args, slide_name, coords, feats, orig_sizes, scalings, model_dicts)


if __name__ == "__main__":
    parser = Options()
    args = parser.parse()

    main(args)
