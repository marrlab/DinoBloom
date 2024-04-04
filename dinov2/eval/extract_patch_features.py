import argparse
import os
from pathlib import Path

import h5py
import torch
import tqdm
from models.return_model import get_models, get_transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser(description="Feature extraction")

parser.add_argument(
    "--model_name",
    help="name of model",
    type=str,
)
# ctranspath
# resnet50
# resnet50_full
# owkin
# dinov2_vits14
# dinov2_vitb14
# dinov2_vitl14
# dinov2_vitg14

parser.add_argument(
    "--dataset",
    help="name of dataset",
    default="wbc_mil_Dataset",
    type=str,
)
parser.add_argument(
    "--train_image_path",
    help="path to folder with images",
    default="/lustre/groups/labs/marr/qscd01/datasets/210526_mll_mil_pseudonymized/splitted_data/train",
    type=str,
)
parser.add_argument(
    "--test_image_path",
    help="path to folder with images",
    default="/lustre/groups/labs/marr/qscd01/datasets/210526_mll_mil_pseudonymized/splitted_data/test",
    type=str,
)

parser.add_argument(
    "--checkpoint",
    help="path to checkpoint",
    default=None,
    type=str,
)
# ctranspath:           “/lustre/groups/shared/users/peng_marr/HistoDINO/logs/baseline_models/ctranspath/ctranspath.pth”
# resnet50:             None
# resnet50_full:        None
# owkin:                None
# dinov2_vits14:        None
# dinov2_vitb14:        None
# dinov2_vitl14:        None
# dinov2_vitg14:        None
# dinov2_vits14 bloom:  "/lustre/groups/shared/users/peng_marr/HistoDINO/models/vits_9999.pth"
# dinov2_vitb14 bloom:  "/lustre/groups/shared/users/peng_marr/HistoDINO/models/vitb_hema_16999.pth"
# dinov2_vitl bloom:    "/lustre/groups/shared/users/peng_marr/HistoDINO/models/vitl_4999_final.pth"
# dinov2_vitg bloom:    "/lustre/groups/shared/users/peng_marr/HistoDINO/models/vitg_4999_final.pth"

parser.add_argument(
    "--save_dir",
    "--save-dir",
    "-s",
    help="path save directory",
    type=str,
)
# ctranspath:           ".../ctranspath"
# resnet50:             ".../resnet50"
# resnet50_full:        ".../resnet50_full"
# owkin:                ".../owkin"
# dinov2_vits14 orig:   ".../dinov2_vits"
# dinov2_vitb14 orig:   ".../dinov2_vitb"
# dinov2_vitl14 orig:   ".../dinov2_vitl"
# dinov2_vitg14 orig:   ".../dinov2_vitg"
# dinov2_vits14 bloom:  ".../dinov2_vits14_bloom"
# dinov2_vitb14 bloom:  ".../dinov2_vitb14_bloom"
# dinov2_vitl14 bloom:  ".../dinov2_vitl14_bloom"
# dinov2_vitg14 bloom:  ".../dinov2_vitg14_bloom"

parser.add_argument(
    "--model_path",
    help="path of model checkpoint",
    default=None,
    type=str,
)


class CustomImageDataset(Dataset):
    def __init__(self, images, transform):
        self.transform = transform
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        image = Image.open(image_path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)

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
                    if cell.lower().endswith(".tif"):
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


def save_features_and_labels_individual(feature_extractor, dataloader, save_dir, args):

    folder_name = args.model_name
    if args.model_name.split("_")[0] == "dinov2" and args.checkpoint is not None:
        folder_name += "_bloom"

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        feature_extractor.eval()

        for images, image_paths in tqdm.tqdm(dataloader):
            images = images.to(device)
            batch_features = feature_extractor(images)

            for img_name, img_features in zip(image_paths, batch_features):

                img_name = img_name.replace("/splitted_data/", f"/splitted_extracted_features/{folder_name}/")
                h5_filename = f"{img_name.split('.')[0]}.h5"

                os.makedirs(os.path.dirname(h5_filename), exist_ok=True)

                with h5py.File(h5_filename, "w") as hf:
                    hf.create_dataset("features", data=img_features.cpu().numpy())


def main(args):
    train_image_paths = args.train_image_path
    test_image_paths = args.test_image_path
    model_name = args.model_name
    transform = get_transforms(model_name)

    train_dataset = wbc_mil_Dataset(transform=transform, data_path=train_image_paths)
    test_dataset = wbc_mil_Dataset(transform=transform, data_path=test_image_paths)

    # Create data loaders for the three datasets
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16)

    feature_extractor = get_models(model_name, saved_model_path=args.checkpoint)

    if args.checkpoint is not None:
        model_name = f"{model_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
    args.save_dir = Path(args.save_dir)

    save_features_and_labels_individual(feature_extractor, train_dataloader, os.path.join(args.save_dir, "train"), args)
    save_features_and_labels_individual(feature_extractor, test_dataloader, os.path.join(args.save_dir, "test"), args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
