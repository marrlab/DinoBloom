import argparse
import os
from pathlib import Path

import h5py
import pandas as pd
import torch
import tqdm
from models.return_model import get_models, get_transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Feature extraction")

parser.add_argument(
    "--model_name",
    help="name of model",
    default="dinov2_finetuned",
    type=str,
)
parser.add_argument(
    "--dataset",
    help="name of dataset",
    default="MIL-MLL",
    type=str,
)
parser.add_argument(
    "--image_path_train",
    help="path to csv file",
    default="./dinov2/eval/miccai/bild_pfade_with_label.csv",
    type=str,
)
parser.add_argument(
    "--image_path_test",
    help="path to csv file",
    default="./dinov2/eval/miccai/bild_pfade_with_label_test.csv",
    type=str,
)
parser.add_argument(
    "--checkpoint",
    help="path to checkpoint",
    default=None,
    type=str,
)
parser.add_argument(
    "--save_dir",
    "--save-dir",
    "-s",
    help="path save directory",
    default="/lustre/groups/shared/users/peng_marr/HistoDINO/features",
    type=str,
)
parser.add_argument(
    "--model_path",
    help="path of model checkpoint",
    default="/lustre/groups/shared/histology_data/models/benedikt_nct_baseline_vits.pth",
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
    transform = get_transforms(model_name)
    dataset = CustomImageDataset(df_test, transform=transform, class_to_label=class_to_label)

    # Create data loaders for the three datasets
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)
    feature_extractor = get_models(model_name, saved_model_path=args.checkpoint)

    if args.checkpoint is not None:
        model_name = f"{model_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
    args.save_dir = Path(args.save_dir) / args.dataset / model_name

    save_features_and_labels_individual(feature_extractor, train_dataloader, os.path.join(args.save_dir, "features"))



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
