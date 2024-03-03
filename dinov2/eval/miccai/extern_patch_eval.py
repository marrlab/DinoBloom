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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, log_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import PathImageDataset, create_label_mapping_from_paths

import wandb

parser = argparse.ArgumentParser(description="Feature extraction")
os.environ["WANDB__SERVICE_WAIT"] = "300"


parser.add_argument(
    "--model_name",
    help="name of model",
    default="dinov2_vits14",
    type=str,
)

parser.add_argument(
    "--experiment_name",
    help="name of experiment",
    default="acevedo_rankloss",
    type=str,
)

parser.add_argument(
    "--filetype",
    help="name of file endings, matek=.tiff, acevedo .jpg, mll .TIF",
    default=[".jpg",".tiff",".TIF"],  # Set the default as a list with a single item
    type=str,
    nargs='*',  # This allows multiple values for --filetype
)

parser.add_argument(
    "--num_workers",
    help="num workers to load data",
    default=16,
    type=int,
)

parser.add_argument(
    "--batch_size",
    default=64,
    type=int,
)

# acevedo: /lustre/groups/labs/marr/qscd01/datasets/Acevedo_20/PBC_dataset_normal_DIB/
#matek: /lustre/groups/labs/marr/qscd01/datasets/191024_AML_Matek/AML-Cytomorphology_LMU/
parser.add_argument(
    "--train_dataset",
    help="path to datasetfolder",
    default="/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped",
    type=str,
)

parser.add_argument(
    "--test_datasets",
    help="path to datasetfolder",
    default=["/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Matek_cropped/",
             "/lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/MLL_20221220/"],
    type=str,
    nargs='*',
)

parser.add_argument(
    "--model_path",
    help="path to run directory with models inside",
    default="/home/icb/valentin.koch/dinov2/vits_hema_v2_rankloss_1708872068.641858/eval",
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

parser.add_argument(
    "--wandb_project",
    "--wandb-project",
    "--wandb",
    help="name of wandb project",
    default="dino_eval",
    type=str,
)


def save_features_and_labels(feature_extractor, dataloader, save_dir,dataset_len):

    if dataset_len==len(list(Path(save_dir).glob("*.h5"))):
        print("features already extracted")
        return
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


def create_stratified_folds(labels):
    """
    Splits indices into 5 stratified folds based on the provided labels,
    returning indices for train and test sets for each fold.
    
    Args:
    - labels (array-like): Array or list of labels to be used for creating stratified folds.
    
    Returns:
    - A list of tuples, each containing two arrays: (train_indices, test_indices) for each fold.
    """
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5)
    
    # Prepare for stratified splitting
    folds = []
    for train_index, test_index in skf.split(X=np.arange(len(labels)), y=labels):
        folds.append((train_index, test_index))
    
    return folds

def main(args):

    model_name = args.model_name
    transform = get_transforms(model_name)

    # make sure encoding is always the same

    wandb.init(
    entity="histo-collab",
    project=args.wandb_project,
    name= model_name +"_" +args.experiment_name,
    config=args
    )

    # If you want to log the results with Weights & Biases (wandb), you can initialize a wandb run:


    # sorry for the bad naming here, its not yet sorted :)
    

    if model_name in ["owkin","resnet50","resnet50_full","remedis","imagebind"]:
        sorted_paths=[None]
    elif model_name in ["retccl","ctranspath"]:
        sorted_paths=[Path(args.model_path)]
    else:
        sorted_paths = list(Path(args.model_path).rglob("*teacher_checkpoint.pth"))

    if len(sorted_paths)>1:
        sorted_paths = sorted(sorted_paths, key=sort_key)
    if args.evaluate_untrained_baseline:
        sorted_paths.insert(0, None)

    

    for checkpoint in sorted_paths:
        if checkpoint is not None:
            parent_dir=checkpoint.parent 
        else:
            parent_dir = Path(args.model_path) / (model_name+"_baseline")
            
        print("loading checkpoint: ", checkpoint)
        feature_extractor = get_models(model_name, saved_model_path=checkpoint)
        dataset_paths=[args.train_dataset,*args.test_datasets]

        all_dataset_features=[]
        all_dataset_labels=[]

        create_label_mapping_from_paths(Path(args.train_dataset).glob("*"+args.filetype[0]))

        for i,dataset_path in enumerate(dataset_paths):
            dataset_name=Path(dataset_path).stem
            feature_dir = parent_dir / args.experiment_name / (dataset_name+"_features")
            
            dataset = PathImageDataset(dataset_path, transform=transform, filetype=args.filetype[i])
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            save_features_and_labels(feature_extractor, dataloader, feature_dir, len(dataset))
    
            all_features=list(feature_dir.glob("*.h5"))
            features,labels=get_data(all_features)
            all_dataset_features.append(features)
            all_dataset_labels.append(labels)

        log_reg_folds=[]
        knn_folds_1=[]
        knn_folds_20=[]

        train_dataset_labels=all_dataset_labels[0]
        train_dataset_features=all_dataset_features[0]

        folds=create_stratified_folds(train_dataset_labels)

        for i, (train_indices, test_indices) in enumerate(folds):
            assert not set(train_indices) & set(test_indices), "There are common indices in train and test lists."

            train_data=train_dataset_labels[train_indices]
            train_labels=train_dataset_features[train_indices]

            test_data_in_domain=train_dataset_features[test_indices]
            test_labels_in_domain=train_dataset_labels[test_indices]
            # Create data loaders for the  datasets

            print("data fully loaded")
            joined_test_data=[test_data_in_domain,*all_dataset_features[1:]]
            joined_test_labels=[test_labels_in_domain,*all_dataset_labels[1:]]
            if args.logistic_regression:
                logreg_dir = parent_dir/ "log_reg_eval"
                log_reg = train_and_evaluate_logistic_regression(
                    train_data, train_labels, joined_test_data,joined_test_labels, logreg_dir, max_iter=1000
                )

                log_reg_folds.append(log_reg)
                print("logistic_regression done")

            if args.umap:
                umap_dir = parent_dir/ "umaps"
                umap_train = create_umap(train_data, train_labels, umap_dir)
                umap_test = create_umap(joined_test_data, joined_test_labels, umap_dir, "test")
                print("umap done")

            if args.knn:
                knn_dir = parent_dir / "knn_eval"
                knn_metrics_1 = perform_knn(train_data, train_labels, joined_test_data, joined_test_labels, knn_dir,1)
                knn_folds_1.append(knn_metrics_1)
                knn_metrics_20 = perform_knn(train_data, train_labels, joined_test_data, joined_test_labels, knn_dir,20)
                knn_folds_20.append(knn_metrics_20)
                print("knn done")

            if checkpoint is not None and len(sorted_paths)>1:
                step = int(parent_dir.name.split("_")[1])
            else: 
                step=0
            
        aggregated_knn_1=average_dicts(knn_folds_1)
        aggregated_knn_20=average_dicts(knn_folds_20)
        aggregated_log_reg=average_dicts(log_reg_folds)

        wandb.log({"knn1": aggregated_knn_1,"knn20": aggregated_knn_20}, step=step)
        wandb.log(
            {"log_reg": aggregated_log_reg, "umap_test": wandb.Image(umap_test), "umap_train": wandb.Image(umap_train)}, step=step
        )

def average_dicts(fold_dicts):
    #shape of fold_dicts: (5,n)
    aggregated={}
    for i in range(fold_dicts[0]): #loop over evaluated datasets
        filtered_dicts=[fold_dict[i] for fold_dict in fold_dicts]
        for key in filtered_dicts[0]:
            # Extract the list of values for the current key from all dictionaries
            values = [d[key] for d in filtered_dicts]
            
            # Calculate the average and standard deviation
            avg = np.mean(values)
            std_dev = np.std(values)
            
            # Store the results
            aggregated[key] = avg
            aggregated[key+"_std"]=std_dev



def process_file(file_name):
    with h5py.File(file_name, "r") as hf:
        features = torch.tensor(hf["features"][:]).tolist()
        label = int(hf["labels"][()])
    return features, label


# {"Accuracy": accuracy, "Balanced_Acc": balanced_acc, "Weighted_F1": weighted_f1}


def get_data(all_data):
    # Define the directories for train, validation, and test data and labels

    # Load training data into dictionaries
    features, labels = [], []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_name) for file_name in all_data]

        for i, future in tqdm(enumerate(futures), desc="Loading data"):
            feature, label = future.result()
            features.append(feature)
            labels.append(label)

    # Convert the lists to NumPy arrays
    features = np.array(features)
    labels = np.array(labels).flatten()
    # Flatten test_data
    features = features.reshape(features.shape[0], -1)  # Reshape to (n_samples, 384)

    return features, labels


def perform_knn(train_data, train_labels, test_data, test_labels, save_dir,n_neighbors):
    # Define a range of values for n_neighbors to search

    os.makedirs(save_dir, exist_ok=True)

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
    print(f"balanced accuracy: {balanced_acc}")
    print(f"accuracy: {accuracy}")
    print(f"weighted f1: {weighted_f1}")

    metrics = {"accuracy": accuracy, "balanced_accuracy": balanced_acc, "weighted_f1": weighted_f1}

    df_labels_to_save = pd.DataFrame({"True Labels": test_labels, "Predicted Labels": test_predictions})
    filename = f"{Path(save_dir).name}_labels_and_predictions.csv"
    file_path = os.path.join(save_dir, filename)
    # Speichern des DataFrames in der CSV-Datei
    df_labels_to_save.to_csv(file_path, index=False)

    return metrics


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


def train_and_evaluate_logistic_regression(train_data, train_labels, all_test_data, all_test_labels, save_dir, max_iter=1000):
    # Initialize wandb

    M = train_data.shape[1]
    C = len(np.unique(train_labels))
    l2_reg_coef = 100 / (M * C)

    # Initialize the logistic regression model with L-BFGS solver
    logistic_reg = LogisticRegression(C=1 / l2_reg_coef, max_iter=max_iter, multi_class="multinomial", solver="lbfgs")

    logistic_reg.fit(train_data, train_labels)
    predicted_probabilities = logistic_reg.predict_proba(train_data)
    loss = log_loss(train_labels, predicted_probabilities)

    performance_dicts=[]

    for test_features,test_labels in zip(all_test_data,all_test_labels):
    # Evaluate the model on the test data
        test_predictions = logistic_reg.predict(test_features)

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
        print(f"balanced accuracy: {balanced_acc}")
        print(f"accuracy: {accuracy}")
        print(f"weighted f1: {weighted_f1}")
        print(report)

        # Log the final loss, accuracy, and classification report using wandb.log
        final_loss = loss
        performance_dict={
            "Final Loss": final_loss,
            "Accuracy": accuracy,
            "Balanced_Acc": balanced_acc,
            "Weighted_F1": weighted_f1,
        #  "Classification Report": wandb.Table(dataframe=report_df),
        }
        print(performance_dict)
        performance_dicts.append(performance_dict)

    return performance_dicts
    


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
