import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import glob

import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os


def create_label_mapping(df):
    """
    Creates a dictionary mapping each unique class label in the dataframe to an integer.

    Parameters:
    - df: pandas DataFrame containing a column 'Label' with class labels.

    Returns:
    - A dictionary mapping each unique label to an integer, starting from 0.
    """
    # Get unique labels and sort them
    unique_labels = sorted(df["Label"].unique())

    # Create mapping
    label_to_int = {label: index for index, label in enumerate(unique_labels)}

    return label_to_int

def create_label_mapping_from_paths(image_paths):
    """
    Creates a dictionary mapping each unique class label, derived from the parent folder name, to an integer.

    Parameters:
    - image_paths: List of strings, where each string is the file path of an image.

    Returns:
    - A dictionary mapping each unique label to an integer, starting from 0.
    """
    # Extract class labels from parent folder names
    labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]

    # Get unique labels and sort them
    unique_labels = sorted(set(labels))

    # Create mapping
    label_to_int = {label: index for index, label in enumerate(unique_labels)}

    return label_to_int

class CustomImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        self.class_to_label = create_label_mapping(df)
        print(self.class_to_label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, label = self.df.iloc[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        encoded_label = self.class_to_label[label]

        return image, encoded_label, Path(image_path).stem

class PathImageDataset(Dataset):
    def __init__(self, image_path, transform,filetype=".tiff"):
        self.images = list(Path(image_path).rglob("*"+filetype))
        self.transform = transform
        self.class_to_label = create_label_mapping_from_paths(self.images)
        print(self.class_to_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image_path=self.images[i]
        label = Path(image_path).parent.name
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        encoded_label = self.class_to_label[label]

        return image, encoded_label, Path(image_path).stem

def bgr_format(xml_string: str):
    """
    Determine whether the image is in BGR or RGB format based on the PixelType element in the image metadata.

    Args:
    - xml_string: a string representing the image metadata in XML format.

    Returns:
    - A boolean value indicating whether the image is in BGR format (True) or not (False).
    """
    if xml_string == "":
        return False

    root = ET.fromstring(xml_string)
    pixel_type_elem = root.findall(".//PixelType")
    return "bgr" in pixel_type_elem[0].text.lower() if pixel_type_elem is not None else False


def get_driver(extension_name: str):
    """
    Determine the driver to use for opening an image file based on its extension.

    Args:
    - extension_name: a string representing the file extension of the image file.

    Returns:
    - A string representing the driver to use for opening the image file.
    """

    if extension_name in [".tiff", ".tif", ".jpg", ".jpeg", ".png"]:
        return "GDAL"
    elif extension_name == "":
        return "DCM"
    else:
        return extension_name.replace(".", "").upper()


def get_scaling(args: argparse.Namespace, mpp_resolution_slide: float):
    """
    Determine the scaling factor to apply to an image based on the desired resolution in micrometers per pixel and the
    resolution in micrometers per pixel of the slide.

    Args:
    - args: a namespace containing the following attributes:
        - downscaling_factor: a float representing the downscaling factor to apply to the image.
        - resolution_in_mpp: a float representing the desired resolution in micrometers per pixel.
    - mpp_resolution_slide: a float representing the resolution in micrometers per pixel of the slide.

    Returns:
    - A float representing the scaling factor to apply to the image.
    """

    if args.downscaling_factor > 0:
        return args.downscaling_factor
    else:
        return args.resolution_in_mpp / (mpp_resolution_slide * 1e06)


def threshold(patch: np.array, args: argparse.Namespace):
    """
    Determine if a patch of an image should be considered invalid based on the following criteria:
    - The number of pixels with color values above a white threshold and below a black threshold should not exceed
    a certain ratio of the total pixels in the patch.
    - The patch should have significant edges.
    If these conditions are not met, the patch is considered invalid and False is returned.

    Args:
    - patch: a numpy array representing the patch of an image.
    - args: a namespace containing at least the following attributes:
        - white_thresh: a float representing the white threshold value.
        - black_thresh: a float representing the black threshold value.
        - invalid_ratio_thresh: a float representing the maximum ratio of foreground pixels to total pixels in the patch.
        - edge_threshold: a float representing the minimum edge value for a patch to be considered valid.

    Returns:
    - A boolean value indicating whether the patch is valid or not.
    """

    # Count the number of whiteish pixels in the patch
    whiteish_pixels = np.count_nonzero(
        (patch[:, :, 0] > args.white_thresh[0])
        & (patch[:, :, 1] > args.white_thresh[1])
        & (patch[:, :, 2] > args.white_thresh[2])
    )

    # Count the number of black pixels in the patch
    black_pixels = np.count_nonzero(
        (patch[:, :, 0] <= args.black_thresh)
        & (patch[:, :, 1] <= args.black_thresh)
        & (patch[:, :, 2] <= args.black_thresh)
    )
    dark_pixels = np.count_nonzero(
        (patch[:, :, 0] <= args.calc_thresh[0])
        & (patch[:, :, 1] <= args.calc_thresh[1])
        & (patch[:, :, 2] <= args.calc_thresh[2])
    )
    calc_pixels = dark_pixels - black_pixels

    if calc_pixels / (patch.shape[0] * patch.shape[1]) >= 0.05:  # we always want to keep calc in!
        return True

    # Compute the ratio of foreground pixels to total pixels in the patch
    invalid_ratio = (whiteish_pixels + black_pixels) / (patch.shape[0] * patch.shape[1])

    # Check if the ratio exceeds the threshold for invalid patches
    if invalid_ratio <= args.invalid_ratio_thresh:
        # Compute the edge map of the patch using Canny edge detection
        edge = cv2.Canny(patch, 40, 100)

        # If the maximum edge value is greater than 0, compute the mean edge value as a percentage of the maximum value
        if np.max(edge) > 0:
            edge = np.mean(edge) * 100 / np.max(edge)
        else:
            edge = 0

        # Check if the edge value is below the threshold for invalid patches or is NaN
        if (edge < args.edge_threshold) or np.isnan(edge):
            return False
        else:
            return True

    else:
        return False


def save_tile_preview(args, slide_name, scn, wsi, coords, tile_path):
    """
    Save the tile preview image with the specified size.

    Args:
        args (argparse.Namespace): A Namespace object that contains the arguments passed to the script.
        slide_name (str): A string representing the name of the slide file.
        scn (int): An integer representing the scene number.
        wsi (numpy.ndarray): A NumPy array representing the whole slide image.
        coords (pandas.DataFrame): A Pandas DataFrame containing the coordinates of the tiles.
        tile_path (pathlib.Path): A Path object representing the path where the tile preview image will be saved.

    Returns:
        None
    """

    # Draw bounding boxes for each tile on the whole slide image
    def draw_rect(wsi, x, y, size, color=[0, 0, 0], thickness=4):
        x2, y2 = x + size, y + size
        wsi[y : y + thickness, x : x + size, :] = color
        wsi[y : y + size, x : x + thickness, :] = color
        wsi[y : y + size, x2 - thickness : x2, :] = color
        wsi[y2 - thickness : y2, x : x + size, :] = color

    for _, [scene, x, y] in coords.iterrows():
        if scn == scene:
            draw_rect(wsi, y, x, args.patch_size)
        # cv2.rectangle(wsi.copy(), (x1, y1), (x2, y2), (0,0,0), thickness=4)

    # Convert NumPy array to PIL Image object
    preview_im = Image.fromarray(wsi)

    # Determine new dimensions of the preview image while maintaining aspect ratio
    preview_size = int(args.preview_size)
    width, height = preview_im.size
    aspect_ratio = height / width

    if aspect_ratio > 1:
        new_height = preview_size
        new_width = int(preview_size / aspect_ratio)
    else:
        new_width = preview_size
        new_height = int(preview_size * aspect_ratio)

    # Resize the preview image
    preview_im = preview_im.resize((new_width, new_height))

    # Save the preview image to disk
    preview_im.save(tile_path / f"{slide_name}_{scn}.png")


def save_qupath_annotation(
    args: argparse.Namespace, slide_name: str, scn: int, coords: pd.DataFrame, annotation_path: str
):
    """
    Saves the QuPath annotation to a geojson file.

    Args:
        args (Namespace): Arguments for the script.
        slide_name (str): The name of the slide.
        scn (int): The SCN number of the slide.
        coords (pandas.DataFrame): The coordinates for the patches.
        annotation_path (pathlib.Path): The path to the output directory.

    Returns:
        None
    """

    # Function to create a single annotation feature
    def create_feature(coordinates, color: str):
        # Define the coordinates of the feature polygon
        x, y = coordinates[0], coordinates[1]
        top_left = coordinates
        top_right = [coordinates[0] + args.patch_size, coordinates[1]]
        bottom_right = [
            coordinates[0] + args.patch_size,
            coordinates[1] + args.patch_size,
        ]
        bottom_left = [coordinates[0], coordinates[1] + args.patch_size]
        coordinates = [top_left, top_right, bottom_right, bottom_left, top_left]

        # Create the feature dictionary with the specified properties
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coordinates]},
            "properties": {
                "objectType": "annotation",
                "classification": {"name": f"{x}, {y}", "color": color},  # random name
            },
        }
        return feature

    # Function to create a feature collection from a list of features
    def create_feature_collection(features):
        feature_collection = {"type": "FeatureCollection", "features": features}
        return feature_collection

    # Define the color of the annotation features
    color = [255, 0, 0]

    # Create a list of annotation features from the provided coordinates
    features = [create_feature([x, y], color) for _, [_, x, y] in coords.iterrows()]

    # Convert the list of features into a feature collection
    features = create_feature_collection(features)

    # Write the feature collection to a GeoJSON file
    with open(annotation_path / f"{slide_name}_{scn}.geojson", "w") as annotation_file:
        # Write the dictionary to the file in JSON format
        json.dump(features, annotation_file)


def save_hdf5(
    save_dir: str,
    args: argparse.Namespace,
    slide_name: str,
    coords: pd.DataFrame,
    feats: dict,
    slide_sizes: list[tuple],
    downscaling_factors: list,
    model_dicts: list[dict],
):
    """
    Save the extracted features and coordinates to an HDF5 file.
    Args:
        args (argparse.Namespace): Arguments containing various processing parameters.
        slide_name (str): Name of the slide file.
        coords (pd.DataFrame): Coordinates of the extracted patches.
        feats (dict): dictionary: modelname: extracted features
    Returns:
        None
    """
    for (model_name, features), model_dict in zip(feats.items(), model_dicts):

        if len(features) > 0:
            with h5py.File(
                Path(save_dir) / f"{slide_name}.h5",
                "w",
            ) as f:
                f["coords"] = coords.astype("float64")
                f["feats"] = features
                f["args"] = json.dumps(vars(args))
                f["model_name"] = model_name
                f["slide_sizes"] = slide_sizes
                f["donwscaling_factor"] = downscaling_factors

            if len(np.unique(coords.scn)) != len(slide_sizes):
                print(
                    "SEMIWARNING, at least for one scene of ",
                    slide_name,
                    "no features were extracted, reason could be poor slide quality.",
                )
        else:
            print(
                "WARNING, no features extracted at slide",
                slide_name,
                "reason could be poor slide quality.",
            )
