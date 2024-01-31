import argparse
import os

import h5py
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
from utils import CustomImageDataset, create_datasets

from models.return_model import get_models, get_transforms



parser = argparse.ArgumentParser(description="Feature extraction")


parser.add_argument(
    "--model_name",
    help="name of model",
    default="dinov2_finetuned",
    type=str,
)
parser.add_argument(
    "--image_path_train",
    help="path to csv file",
    default="/home/icb/valentin.koch/dinov2/dinov2/eval/miccai/bild_pfade_with_label.csv",
    type=str,
)
parser.add_argument(
    "--image_path_test",
    help="path to csv file",
    default="/home/icb/valentin.koch/dinov2/dinov2/eval/miccai/bild_pfade_with_label_test.csv",
    type=str,
)
parser.add_argument(
    "--save_dir",
    help="path save directory",
    default="/lustre/groups/shared/histology_data/features_NCT-CRC-100k-nonorm/benedikt_baseline_vits",
    type=str,
)
parser.add_argument(
    "--model_path",
    help="path of model checkpoint",
    default="/lustre/groups/shared/histology_data/models/benedikt_nct_baseline_vits.pth",
    type=str,
)


def save_features_and_labels_individual(feature_extractor, dataloader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        feature_extractor.eval()

        for images, labels, names in tqdm.tqdm(dataloader):
            images = images.to(device)
            batch_features = feature_extractor(images)

            labels_np = labels.numpy()

            for img_name, img_features, img_label in zip(names, batch_features, labels_np):
                h5_filename = os.path.join(save_dir, f"{img_name}.h5")

                with h5py.File(h5_filename, "w") as hf:
                    hf.create_dataset("features", data=img_features.cpu().numpy())
                    hf.create_dataset("labels", data=img_label)


def main(args):
    image_paths = args.image_path_train
    image_test_paths = args.image_path_test
    model_name = args.model_name
    df = pd.read_csv(image_paths)
    df_test = pd.read_csv(image_test_paths)

    transform = get_transforms(model_name)

    # make sure encoding is always the same
    class_to_label = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}

    train_dataset, val_dataset = create_datasets(df, transform, class_to_label=class_to_label)
    test_dataset = CustomImageDataset(df_test, transform=transform, class_to_label=class_to_label)

    # Create data loaders for the three datasets
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=5)

    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=5)

    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=5)

    feature_extractor = get_models(model_name, args.model_path)

    save_features_and_labels_individual(feature_extractor, train_dataloader, os.path.join(args.save_dir, "train_data"))
    save_features_and_labels_individual(feature_extractor, val_dataloader, os.path.join(args.save_dir, "val_data"))
    save_features_and_labels_individual(feature_extractor, test_dataloader, os.path.join(args.save_dir, "test_data"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
