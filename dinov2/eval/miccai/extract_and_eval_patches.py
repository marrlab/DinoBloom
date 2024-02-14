import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from models.return_model import get_models, get_transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import CustomImageDataset

import wandb

parser = argparse.ArgumentParser(description="Feature extraction")
os.environ["WANDB__SERVICE_WAIT"] = "300"


parser.add_argument(
    "--model_name",
    help="name of model",
    default="dinov2_finetuned",
    type=str,
)

parser.add_argument(
    "--experiment_name",
    help="name of experiment",
    default="crc_patch",
    type=str,
)

parser.add_argument(
    "--num_workers",
    help="num workers to load data",
    default=16,
    type=int,
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
    "--run_path",
    help="path to run directory with models inside",
    default="/home/icb/valentin.koch/dinov2/debug/eval",
    type=str,
)

parser.add_argument(
    "--knn",
    help="perform knn or not",
    default=True,
    type=bool,
)

parser.add_argument(
    "--evaluate_untrained_baseline",
    help="Set to true if original dino should be tested.",
    action='store_true',
)

parser.add_argument(
    "--logistic_regression",
    "--logistic-regression",
    "-log",
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


def save_features_and_labels_individual(feature_extractor, dataloader, save_dir):
    print("extracting features..")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        feature_extractor.eval()

        for images, labels, names in tqdm(dataloader):
            images = images.to(device)
            batch_features = feature_extractor(images)

            labels_np = labels.numpy()

            for img_name, img_features, img_label in zip(names, batch_features, labels_np):
                h5_filename = os.path.join(save_dir, f"{img_name}.h5")

                with h5py.File(h5_filename, "w") as hf:
                    hf.create_dataset("features", data=img_features.cpu().numpy())
                    hf.create_dataset("labels", data=img_label)


def sort_key(path):
    # Extract the numeric part from the directory name
    # Assuming the format is always like '.../train_xxxx/...'
    number_part = int(path.parts[-2].split("_")[1])
    return number_part


def main(args):
    image_paths = args.image_path_train
    image_test_paths = args.image_path_test
    model_name = args.model_name

    df = pd.read_csv(image_paths)
    df_test = pd.read_csv(image_test_paths)

    transform = get_transforms(model_name)

    # make sure encoding is always the same

    train_dataset = CustomImageDataset(df, transform=transform)
    test_dataset = CustomImageDataset(df_test, transform=transform)

    # Create data loaders for the  datasets
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=args.num_workers)

    # If you want to log the results with Weights & Biases (wandb), you can initialize a wandb run:
    wandb.init(
        entity="histo-collab",
        project="dino_eval",
        name= model_name +"_" +args.experiment_name,
    )

    # sorry for the bad naming here, its not yet sorted :)
    sorted_paths = list(Path(args.run_path).rglob("*.pth"))
    if len(sorted_paths)>1:
        sorted_paths = sorted(sorted_paths, key=sort_key)
    if args.evaluate_untrained_baseline:
        sorted_paths.insert(0, None)

    for checkpoint in sorted_paths:
        if checkpoint is not None:
            parent_dir=checkpoint.parent 
        else:
            parent_dir = Path(args.run_path) / (model_name+"_baseline")

        feature_extractor = get_models(model_name, saved_model_path=checkpoint)
        feature_dir = parent_dir / args.experiment_name

        train_dir = os.path.join(feature_dir, "train_data")
        test_dir = os.path.join(feature_dir, "test_data")

        save_features_and_labels_individual(feature_extractor, train_dataloader, train_dir)
        save_features_and_labels_individual(feature_extractor, test_dataloader, test_dir)

        train_data, train_labels, test_data, test_labels = get_data(train_dir, test_dir)

        print("data fully loaded")
        print("Shape of train_data:", train_data.shape)
        print("Shape of train_labels:", train_labels.shape)
        print("Shape of test_data:", test_data.shape)
        print("Shape of test_labels:", test_labels.shape)

        if args.logistic_regression:
            logreg_dir = parent_dir/ "log_reg_eval"
            log_reg = train_and_evaluate_logistic_regression(
                train_data, train_labels, test_data, test_labels, logreg_dir, max_iter=1000
            )
            print("logistic_regression done")

        if args.umap:
            umap_dir = parent_dir/ "umaps"
            umap_train = create_umap(train_data, train_labels, umap_dir)
            umap_test = create_umap(test_data, test_labels, umap_dir, "test")
            print("umap done")

        if args.knn:
            knn_dir = parent_dir / "knn_eval"
            knn_metrics = perform_knn(train_data, train_labels, test_data, test_labels, knn_dir)
            print("knn done")

        if checkpoint is not None and len(sorted_paths)>1:
            step = int(parent_dir.name.split("_")[1])
        else: 
            step=0

        wandb.log(knn_metrics, step=step)
        wandb.log(
            {"log_reg": log_reg, "umap_test": wandb.Image(umap_test), "umap_train": wandb.Image(umap_train)}, step=step
        )


def process_file(file_name):
    with h5py.File(file_name, "r") as hf:
        # hf.visititems(print)
        features = torch.tensor(hf["features"][:]).tolist()
        label = int(hf["labels"][()])
    return features, label


# {"Accuracy": accuracy, "Balanced_Acc": balanced_acc, "Weighted_F1": weighted_f1}


def get_data(train_dir, test_dir):
    # Define the directories for train, validation, and test data and labels

    # Load training data into dictionaries
    train_features, train_labels = [], []
    test_features, test_labels = [], []

    train_files = list(Path(train_dir).glob("*.h5"))
    test_files = list(Path(test_dir).glob("*.h5"))

    with ThreadPoolExecutor() as executor:
        futures_train = [executor.submit(process_file, file_name) for file_name in train_files]

        for i, future_train in tqdm(enumerate(futures_train), desc="Loading training data"):
            feature_train, label_train = future_train.result()
            train_features.append(feature_train)
            train_labels.append(label_train)

    with ThreadPoolExecutor() as executor:
        futures_test = [executor.submit(process_file, file_name) for file_name in test_files]

        for i, future in tqdm(enumerate(futures_test), desc="Loading test data"):
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


def perform_knn(train_data, train_labels, test_data, test_labels, save_dir):
    # Define a range of values for n_neighbors to search
    n_neighbors_values = [1, 20]
    # n_neighbors_values = [1, 2, 5, 10, 20, 50, 100, 500]
    # n_neighbors_values = [1, 2, 3, 4, 5] # -> for testing
    metrics_dict = {}
    os.makedirs(save_dir, exist_ok=True)

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
        weighted_f1 = f1_score(test_labels, test_predictions, average="weighted")

        print(f"n_neighbors = {n_neighbors}")
        print(f"Accuracy: {accuracy}")

        ## Calculate the classification report
        report = classification_report(test_labels, test_predictions, output_dict=True)

        print(f"report: {report}")

        current_metrics = {"accuracy": accuracy, "balanced_accuracy": balanced_acc, "weighted_f1": weighted_f1}

        # Store the metrics dictionary in the metrics_dict with a key indicating the number of neighbors
        metrics_dict[f"knn_{n_neighbors}"] = current_metrics
        # Convert the report to a Pandas DataFrame for logging
        # report_df = pd.DataFrame(report).transpose()

        # Log the final loss, accuracy, and classification report using wandb.log
        # wandb.log({"Classification Report": wandb.Table(dataframe=report_df)})

        df_labels_to_save = pd.DataFrame({"True Labels": test_labels, "Predicted Labels": test_predictions})
        filename = f"{Path(save_dir).name}_labels_and_predictions.csv"
        file_path = os.path.join(save_dir, filename)
        # Speichern des DataFrames in der CSV-Datei
        df_labels_to_save.to_csv(file_path, index=False)

    return metrics_dict


def create_umap(data, labels, save_dir, filename_addon="train"):
    # Create a UMAP model and fit it to your data
    # reducer = umap.UMAP(random_state=42)
    reducer = umap.UMAP()
    umap_data = reducer.fit_transform(data)

    # Specify the directory for saving the images

    umap_dir = os.path.join(save_dir, "umaps")
    os.makedirs(umap_dir, exist_ok=True)

    # Loop through different figure sizes
    size = (12, 8)  # Add more sizes as needed

    # Create a scatter plot with the specified size
    plt.figure(figsize=size, dpi=300)
    plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=0.1, cmap="Spectral")
    plt.colorbar()
    plt.title("UMAP")

    # Specify the filename with the size information
    image_filename = f"umap_visualization_{Path(save_dir).name}_{size[0]}x{size[1]}_{filename_addon}.png"

    # Save the UMAP visualization as an image in the specified directory
    plt.savefig(os.path.join(umap_dir, image_filename))
    im = Image.open(os.path.join(umap_dir, image_filename))
    return im


def train_and_evaluate_logistic_regression(train_data, train_labels, test_data, test_labels, save_dir, max_iter=1000):
    # Initialize wandb

    M = train_data.shape[1]
    C = len(np.unique(train_labels))
    l2_reg_coef = 100 / (M * C)

    # Initialize the logistic regression model with L-BFGS solver
    logistic_reg = LogisticRegression(C=1 / l2_reg_coef, max_iter=max_iter, multi_class="multinomial", solver="lbfgs")

    logistic_reg.fit(train_data, train_labels)

    # Evaluate the model on the test data
    test_predictions = logistic_reg.predict(test_data)
    predicted_probabilities = logistic_reg.predict_proba(train_data)
    loss = log_loss(train_labels, predicted_probabilities)
    accuracy = accuracy_score(test_labels, test_predictions)
    balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
    weighted_f1 = f1_score(test_labels, test_predictions, average="weighted")
    # auroc = roc_auc_score(test_labels, test_predictions, multi_class='ovr', average='weighted')
    report = classification_report(test_labels, test_predictions, output_dict=True)

    df_labels_to_save = pd.DataFrame({"True Labels": test_labels, "Predicted Labels": test_predictions})
    filename = f"{Path(save_dir).name}_labels_and_predictions.csv"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    # Speichern des DataFrames in der CSV-Datei
    df_labels_to_save.to_csv(file_path, index=False)

    predicted_probabilities_df = pd.DataFrame(
        predicted_probabilities, columns=[f"Probability Class {i}" for i in range(predicted_probabilities.shape[1])]
    )
    predicted_probabilities_filename = f"{Path(save_dir).name}_predicted_probabilities_test.csv"
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
    return {
        "Final Loss": final_loss,
        "Accuracy": accuracy,
        "Balanced_Acc": balanced_acc,
        "Weighted_F1": weighted_f1,
        "Classification Report": wandb.Table(dataframe=report_df),
    }


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
