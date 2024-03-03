import argparse
import os
from pathlib import Path

import h5py
import pandas as pd
import torch
import tqdm
from models.return_model import get_models, get_transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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
    help="path to checkpoint",
    default=None, #"/lustre/groups/shared/users/peng_marr/pretrained_models/owkin.pth", None
    type=str,
)
parser.add_argument(
    "--save_dir",
    "--save-dir",
    "-s",
    help="path save directory",
    default="/lustre/groups/labs/marr/qscd01/datasets/210526_mll_mil_pseudonymized/splitted_extracted_features/dinov2_vitb_orig/test",
    type=str,
)
parser.add_argument(
    "--model_path",
    help="path of model checkpoint",
    default=None,
    type=str,
)

class CustomImageDataset(Dataset):
    def __init__(self, images, transform):
        self.transform = transform
        self.images=images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        image = Image.open(image_path).convert("RGB").resize((224,224),Image.Resampling.LANCZOS)

        if self.transform:
            image = self.transform(image)
        return image, Path(image_path).name
    

class wbc_mil_Dataset(Dataset):
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
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_path
    

def save_features_and_labels_individual(feature_extractor, dataloader, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        feature_extractor.eval()

        for images, image_paths in tqdm.tqdm(dataloader):
            images = images.to(device)
            batch_features = feature_extractor(images)

            for img_name, img_features in zip(image_paths, batch_features):
                img_name = img_name.replace('/splitted_data/', '/splitted_extracted_features/dinov2_vitb_orig/')
                h5_filename = f"{img_name.split('.')[0]}.h5"

                os.makedirs(os.path.dirname(h5_filename), exist_ok=True)

                with h5py.File(h5_filename, "w") as hf:
                    hf.create_dataset("features", data=img_features.cpu().numpy())


def main(args):
    image_paths = args.image_path
    model_name = args.model_name
    transform = get_transforms(model_name)
    dataset = wbc_mil_Dataset(transform=transform, data_path=image_paths)

    # Create data loaders for the three datasets
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)
    feature_extractor = get_models(model_name, saved_model_path=args.checkpoint)

    if args.checkpoint is not None:
        model_name = f"{model_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
    args.save_dir = Path(args.save_dir)

    save_features_and_labels_individual(feature_extractor, dataloader, os.path.join(args.save_dir))



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
