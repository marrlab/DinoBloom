from torch.utils.data import random_split
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pandas as pd  # Make sure to import pandas
from pathlib import Path

class CustomImageDataset(Dataset):
    def __init__(self, df, transform, class_to_label):
        self.df = df
        self.transform = transform
        self.class_to_label = class_to_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, label = self.df.iloc[idx]

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        encoded_label = self.class_to_label[label]

        return image, encoded_label, Path(image_path).stem

from sklearn.model_selection import train_test_split

def create_datasets(data, transform, class_to_label):
    train_val_idxs, val_idxs = train_test_split(
        range(len(data)), 
        test_size=0.10,    
        stratify=data['Label'],  
        random_state=42,
    )

    # Create training, validation, and test datasets
    train_dataset = CustomImageDataset(
        data.iloc[train_val_idxs],
        transform=transform,
        class_to_label=class_to_label
    )
    
    val_dataset = CustomImageDataset(
        data.iloc[val_idxs],
        transform=transform,
        class_to_label=class_to_label
    )

    return train_dataset, val_dataset

