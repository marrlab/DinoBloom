from model import MyModel
from utils import CustomImageDataset, create_datasets
import os
import torch
from pathlib import Path
import yaml
import numpy as np
import h5py 

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import wandb
import pandas as pd
from PIL import Image
from torchvision import transforms

from utils import CustomImageDataset, create_datasets
from models.return_model import get_models, get_transforms
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import umap
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description="Feature extraction")
os.environ["WANDB__SERVICE_WAIT"] = "300"

parser.add_argument(
    "--knn",
    help="perform knn or not",
    default=True,
    type=bool,
)
parser.add_argument(
    "--logistic_regression",
    help="perform logistic regression or not",
    default=True,
    type=bool,
)
parser.add_argument(
    "--umap",
    help="perform umap or not",
    default=True,
    type=bool,
)
parser.add_argument(
    "--path_folder",
    help="path to folder containing subfolders with training, val and test data",
    default="/lustre/groups/shared/histology_data/features_NCT-CRC-100k-nonorm/dinov2_vit_s_224_baseline_12500/",
    type=str,
)

parser.add_argument(
    "--save_dir",
    help="specify where to save the umap",
    default="/lustre/groups/shared/histology_data/eval/dinoeval",
    type=str,
)
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import h5py
import torch



def process_file(file_name):
    with h5py.File(file_name, 'r') as hf:
        features = torch.tensor(hf['features'][:]).tolist()
        label = int(hf['labels'][()])
    return features, label


def get_data(args):
    # Define the directories for train, validation, and test data and labels
    train_dir = os.path.join(args.path_folder, 'train_data')
    validation_dir = os.path.join(args.path_folder, 'val_data')
    test_dir = os.path.join(args.path_folder, 'test_data')

    # Initialize dictionaries to store data and labels
    data_dict = {}
    labels_dict = {}
    time0=time.time()
    # Load training data into dictionaries
    train_features=[]
    train_labels=[]

    test_features=[]
    test_labels=[]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_name) for file_name in list(Path(train_dir).glob("*.h5"))[:50]]

        for i, future in enumerate(futures):
            if i % 100 == 0:
                print(i)
            features, label = future.result()
            train_features.append(features)
            train_labels.append(label)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_name) for file_name in list(Path(validation_dir).glob("*.h5"))[:50]]

        for i, future in enumerate(futures):
            if i % 100 == 0:
                print(i)
            features, label = future.result()
            test_features.append(features)
            test_labels.append(label)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_name) for file_name in list(Path(test_dir).glob("*.h5"))[:50]]

        for i, future in enumerate(futures):
            if i % 100 == 0:
                print(i)
            features, label = future.result()
            test_features.append(features)
            test_labels.append(label)

    # Convert the lists to NumPy arrays

    test_data = np.array(test_features)
    test_labels = np.array(test_labels).flatten()
    # Flatten test_data
    test_data = test_data.reshape(test_data.shape[0], -1)  # Reshape to (n_samples, 384)
    
    train_data = np.array(train_features)
    train_labels = np.array(train_labels).flatten()
    # Flatten test_data
    train_data = train_data.reshape(train_data.shape[0], -1)

    return train_data, train_labels, test_data, test_labels

def test_data_creation():
    # Create synthetic data
    num_samples = 100  # Number of samples in the synthetic dataset
    num_features = 100  # Number of features per sample

    # Generate random synthetic data
    synthetic_data = np.random.rand(num_samples, num_features)

    # Generate random synthetic labels
    synthetic_labels = np.random.randint(0, 2, size=num_samples)  # Binary classification example

    # Split the synthetic data into a smaller train and test set
    train_data, test_data, train_labels, test_labels = train_test_split(synthetic_data, synthetic_labels, test_size=0.2)

    return train_data, train_labels, test_data, test_labels


def perform_knn(args, train_data, train_labels, test_data, test_labels,save_dir):
    # Define a range of values for n_neighbors to search
    n_neighbors_values = [1, 20]
    #n_neighbors_values = [1, 2, 5, 10, 20, 50, 100, 500]
    #n_neighbors_values = [1, 2, 3, 4, 5] # -> for testing

    for n_neighbors in n_neighbors_values:
        # Initialize a KNeighborsClassifier with the current n_neighbors
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the KNN classifier to the training data
        knn.fit(train_data, train_labels)

        # Predict labels for the test data
        test_predictions = knn.predict(test_data)

        # Evaluate the classifier
        accuracy = accuracy_score(test_labels, test_predictions)
        balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
        weighted_f1 = f1_score(test_labels, test_predictions, average='weighted')

        print(f"n_neighbors = {n_neighbors}")
        print(f"Accuracy: {accuracy}")

        run_name = f"KNN_Training_n_neighbors_{n_neighbors}_{args.path_folder.split('/')[-1]}"

        # If you want to log the results with Weights & Biases (wandb), you can initialize a wandb run:
        wandb.init(project="knn", name=run_name)

        # Log the n_neighbors value, accuracy
        wandb.log({"n_neighbors": n_neighbors, "Accuracy": accuracy, "Balanced_Acc": balanced_acc, "Weighted_F1": weighted_f1})

        ## Calculate the classification report
        report = classification_report(test_labels, test_predictions, output_dict=True)

        print(f"report: {report}")

        # Convert the report to a Pandas DataFrame for logging
        report_df = pd.DataFrame(report).transpose()

        # Log the final loss, accuracy, and classification report using wandb.log
        wandb.log({"Classification Report": wandb.Table(dataframe=report_df)})

        # Finish the wandb run
        wandb.finish()

        df_labels_to_save = pd.DataFrame({'True Labels': test_labels, 'Predicted Labels': test_predictions})
        filename = f"{run_name}_labels_and_predictions.csv"
        file_path = os.path.join(save_dir, filename)
        # Speichern des DataFrames in der CSV-Datei
        df_labels_to_save.to_csv(file_path, index=False)

def create_umap(args, data, labels,save_dir,filename_addon="train"):
    # Create a UMAP model and fit it to your data
    #reducer = umap.UMAP(random_state=42)
    reducer = umap.UMAP()
    umap_data = reducer.fit_transform(data)

    # Specify the directory for saving the images

    umap_dir= os.path.join(save_dir, "umaps")
    os.makedirs(umap_dir, exist_ok=True)

    # Loop through different figure sizes
    figure_sizes = [(48, 32), (36, 24), (24, 16), (12, 8), (6, 4)]  # Add more sizes as needed

    for size in figure_sizes:
        # Create a scatter plot with the specified size
        plt.figure(figsize=size, dpi=300)
        plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=0.1, cmap='Spectral')
        plt.colorbar()
        plt.title("UMAP")

        # Specify the filename with the size information
        image_filename = f'umap_visualization_{args.path_folder.split("/")[-1]}_{size[0]}x{size[1]}_{filename_addon}.png'

        # Save the UMAP visualization as an image in the specified directory
        plt.savefig(os.path.join(umap_dir, image_filename))


def train_and_evaluate_logistic_regression(train_data, train_labels, test_data, test_labels, args, save_dir, max_iter=1000):
    # Initialize wandb
    run_name = f"LogisticRegression_Training_{args.path_folder.split('/')[-1]}"
    wandb.init(project="logistic_regression", name=run_name)

    M = train_data.shape[1]  
    C = 9  
    l2_reg_coef = 100 / (M * C)

    # Initialize the logistic regression model with L-BFGS solver
    logistic_reg = LogisticRegression(
        C=1 / l2_reg_coef,  
        max_iter=max_iter,
        multi_class='multinomial',
        solver='lbfgs'
    )

    logistic_reg.fit(train_data, train_labels)

    # Evaluate the model on the test data
    test_predictions = logistic_reg.predict(test_data)
    predicted_probabilities = logistic_reg.predict_proba(train_data)
    loss = log_loss(train_labels, predicted_probabilities)
    accuracy = accuracy_score(test_labels, test_predictions)
    balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
    weighted_f1 = f1_score(test_labels, test_predictions, average='weighted')
    #auroc = roc_auc_score(test_labels, test_predictions, multi_class='ovr', average='weighted')
    report = classification_report(test_labels, test_predictions, output_dict=True)

    df_labels_to_save = pd.DataFrame({'True Labels': test_labels, 'Predicted Labels': test_predictions})    
    filename = f"{run_name}_labels_and_predictions.csv"
    file_path = os.path.join(save_dir, filename)
    # Speichern des DataFrames in der CSV-Datei
    df_labels_to_save.to_csv(file_path, index=False)

    predicted_probabilities_df = pd.DataFrame(predicted_probabilities, columns=[f'Probability Class {i}' for i in range(predicted_probabilities.shape[1])])
    predicted_probabilities_filename = f"{run_name}_predicted_probabilities_test.csv"
    predicted_probabilities_file_path = os.path.join(save_dir, predicted_probabilities_filename)
    predicted_probabilities_df.to_csv(predicted_probabilities_file_path, index=False)

    # Convert the report to a Pandas DataFrame for logging
    report_df = pd.DataFrame(report).transpose()

    # some prints
    print(f"Final Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print(report)

    # Log the final loss, accuracy, and classification report using wandb.log
    final_loss = loss
    wandb.log({"Final Loss": final_loss, "Accuracy": accuracy, "Balanced_Acc": balanced_acc, "Weighted_F1": weighted_f1, "Classification Report": wandb.Table(dataframe=report_df)})

    # Finish the wandb run
    wandb.finish()



def main(args):
    train_data, train_labels, test_data, test_labels = get_data(args)
    #train_data, train_labels, test_data, test_labels = test_data_creation()
    print("data fully loaded")
    print("Shape of train_data:", train_data.shape)
    print("Shape of train_labels:", train_labels.shape)
    print("Shape of test_data:", test_data.shape)
    print("Shape of test_labels:", test_labels.shape)
    save_directory = os.path.join(args.save_dir, args.path_folder.split("/")[-1])

    if args.logistic_regression:
        train_and_evaluate_logistic_regression(train_data, train_labels, test_data, test_labels, args,save_directory, max_iter=1000)
        print("logistic_regression done")

    if args.umap:
        create_umap(args, train_data, train_labels,save_directory)
        create_umap(args, test_data, test_labels,save_directory,"test")
        print("umap done")

    if args.knn:
        perform_knn(args, train_data, train_labels, test_data, test_labels,save_directory)
        print("knn done")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
