import argparse
import os
from pathlib import Path

import imageio
import h5py
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tqdm
from models.return_model import get_models, get_transforms
from PIL import Image

parser = argparse.ArgumentParser(description="Feature extraction")

parser.add_argument(
    "--model_name",
    help="name of model",
    default="dinov2_vitb14", # ctranspath, resnet50, resnet50_full, owkin, dinov2_vits14, dinov2_vitb14
    type=str,
)
parser.add_argument(
    "--dataset",
    help="name of dataset",
    default="wbc_mil_Dataset",
    type=str,
)
parser.add_argument(
    "--image_path",
    help="path to folder with images",
    default="/lustre/groups/labs/marr/qscd01/datasets/210526_mll_mil_pseudonymized/splitted_data/test",
    type=str,
)

parser.add_argument(
    "--checkpoint",
parser = argparse.ArgumentParser(description="Feature extraction")

parser.add_argument(
    "--model_name",
    help="name of model",
    default="dinov2_vitb14", # ctranspath, resnet50, resnet50_full, owkin, dinov2_vits14, dinov2_vitb14
    type=str,
)
parser.add_argument(
    "--dataset",
    help="name of dataset",
    default="wbc_mil_Dataset",
    type=str,
)
parser.add_argument(
    "--image_path",
    help="path to folder with images",
    default="/lustre/groups/labs/marr/qscd01/datasets/210526_mll_mil_pseudonymized/splitted_data/test",
    type=str,
)

parser.add_argument(
    "--checkpoint",
parser = argparse.ArgumentParser(description="Feature extraction")

parser.add_argument(
    "--model_name",
    help="name of model",
    default="dinov2_vitb14", # ctranspath, resnet50, resnet50_full, owkin, dinov2_vits14, dinov2_vitb14
    type=str,
)
parser.add_argument(
    "--dataset",
    help="name of dataset",
    default="wbc_mil_Dataset",
    type=str,
)
parser.add_argument(
    "--image_path",
    help="path to folder with images",
    default="/lustre/groups/labs/marr/qscd01/datasets/210526_mll_mil_pseudonymized/splitted_data/test",
    type=str,
)

parser.add_argument(
    "--checkpoint",
parser = argparse.ArgumentParser(description="Feature extraction")

parser.add_argument(
    "--model_name",
    help="name of model",
    default="dinov2_vitb14", # ctranspath, resnet50, resnet50_full, owkin, dinov2_vits14, dinov2_vitb14
    type=str,
)
parser.add_argument(
    "--dataset",
    help="name of dataset",
    default="wbc_mil_Dataset",
    type=str,
)
parser.add_argument(
    "--image_path",
    help="path to folder with images",
    default="/lustre/groups/labs/marr/qscd01/datasets/210526_mll_mil_pseudonymized/splitted_data/test",
    type=str,
)

parser.add_argument(
    "--checkpoint",
    help="path to checkpoint",
    default=None, #"/lustre/groups/shared/users/peng_marr/pretrained_models/owkin.pth", None
    type=str,
)
parser.add_argument(
    "--save_dir",
    "--save-dir",
    "-s",
    help="path save directory",
    default="/lustre/groups/shared/users/peng_marr/HistoDINO/test",
    type=str,
)
parser.add_argument(
    "--model_path",
    help="path of model checkpoint",
    default=None,
    type=str,
)
 
 # remedis transforms
def preprocess_image(image, height, width, is_training=False,
                     color_distort=True, test_crop=True):
    """Preprocesses the given image.
    Args:
        image: `Tensor` representing an image of arbitrary size.
        height: Height of output image.
        width: Width of output image.
        is_training: `bool` for whether the preprocessing is for training.
        color_distort: whether to apply the color distortion.
        test_crop: whether or not to extract a central crop of the images
            (as for standard ImageNet evaluation) during the evaluation.
    Returns:
        A preprocessed image `Tensor` of range [0, 1].
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if is_training:
        return preprocess_for_train(image, height, width, color_distort)
    else:
        return preprocess_for_eval(image, height, width, test_crop)

def preprocess_for_eval(image, height, width, crop=True):
    """Preprocesses the given image for evaluation.
    Args:
        image: `Tensor` representing an image of arbitrary size.
        height: Height of output image.
        width: Width of output image.
        crop: Whether or not to (center) crop the test images.
    Returns:
        A preprocessed image `Tensor`.
    """
    if crop:
        image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
    image = tf.image.resize(image, [height, width])  # added by sophia
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image
class WBCMILDataset(tf.keras.utils.Sequence):
    def __init__(self, data_path, transform):
        self.transform = transform
        self.images = []
        
        clses = os.listdir(data_path)
        for cls in clses:
            patients = os.listdir(os.path.join(data_path, cls))
            for patient in patients:
                cells = os.listdir(os.path.join(data_path, cls, patient))
                for cell in cells:
                    if cell.lower().endswith('.tif'):
                        cell_path = os.path.join(data_path, cls, patient, cell)
                        self.images.append(cell_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = imageio.imread(image_path)
        image = preprocess_image(
        image, 224, 224,
        is_training=False, color_distort=False, test_crop=False)
        # image = tf.convert_to_tensor(image, dtype=tf.float32)

        return image, image_paths
    

def save_features_and_labels_individual(feature_extractor, dataloader, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"

    for images, image_paths in tqdm.tqdm(dataloader):
        batch_features = feature_extractor(images)
        batch_features = tf.stop_gradient(batch_features)

        for img_name, img_features in zip(image_paths, batch_features):
            img_name = img_name.numpy().decode("utf-8").replace('/splitted_data/', '/splitted_extracted_features/dinov2_vitb_orig/')
            h5_filename = f"{os.path.splitext(img_name)[0]}.h5"
            h5_filename = Path(save_dir) / Path(h5_filename).name

            os.makedirs(os.path.dirname(h5_filename), exist_ok=True)

            with h5py.File(h5_filename, "w") as hf:
                hf.create_dataset("features", data=img_features.numpy())

def main(args):
    image_paths = args.image_path
    model_name = args.model_name
    transform = get_transforms(model_name)
    dataset = WBCMILDataset(transform=transform, data_path=image_paths)

    batch_size = 8
    num_workers = 16

    dataloader = tf.data.Dataset.from_generator(
        generator=lambda: dataset,
        output_types=(tf.float32, tf.string),
        output_shapes=((None, None, 3), ()))

    dataloader = dataloader.batch(batch_size)
    dataloader = dataloader.prefetch(num_workers)

    feature_extractor = hub.load('/lustre/groups/shared/users/peng_marr/pretrained_models/physionet.org/files/medical-ai-research-foundation/1.0.0/path-50x1-remedis-m')

    if args.checkpoint is not None:
        model_name = f"{model_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
    args.save_dir = Path(args.save_dir)

    save_features_and_labels_individual(feature_extractor, dataloader, os.path.join(args.save_dir))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
