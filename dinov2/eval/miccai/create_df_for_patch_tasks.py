
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="split generation")

parser.add_argument(
    "--dataset_path",
    help="path to dataset",
    default="",
    type=str,
)

parser.add_argument(
    "--dataset_path",
    help="dataset_name",
    default="",
    type=str,
)


def create_train_val_split(folder_path,dataset_name):
    """
    Splits the data in the given folder into training and validation sets,
    ensuring class balance between the sets. Assumes the folder structure
    where the parent folder name is the class label.

    Parameters:
    - folder_path: str, the path to the folder containing the class folders.

    Creates two files:
    - 'train.csv': Contains paths and labels for the training set.
    - 'val.csv': Contains paths and labels for the validation set.
    """
    data = []
    
    # Iterate over each class directory in the folder
    for class_name in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, class_name)
        if os.path.isdir(class_dir):
            # For each image in the class directory
            for img in os.listdir(class_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg',".tif",".tiff",".bmp")):
                    # Collect image path and class name
                    img_path = os.path.join(class_dir, img)
                    data.append((img_path, class_name))
    
    # Convert collected data into a DataFrame
    df = pd.DataFrame(data, columns=['Image Path', 'Label'])
    
    # Split the data ensuring class balance
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
    
    # Save to CSV files
    train_df.to_csv(dataset_name+'_train.csv', index=False)
    val_df.to_csv(dataset_name+'_val.csv', index=False)
    
    return train_df.shape, val_df.shape


if __name__ == "__main__":
    args = parser.parse_args()
    create_train_val_split(args.dataset_path)


