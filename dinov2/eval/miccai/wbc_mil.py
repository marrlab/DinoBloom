import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse


import random
import os
import h5py
import wandb



class WbcMilFeatureDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.features_dict = {}

        # Loop over classes
        clses = os.listdir(data_path)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(clses)}

        for cls in clses:
            cls_path = os.path.join(data_path, cls)
            self.features_dict[cls] = {}
            patients = os.listdir(cls_path)

            # Loop over patients in each class
            for patient in patients:
                patient_path = os.path.join(cls_path, patient)
                cells = os.listdir(patient_path)

                # Initialize an empty list to store features for the patient
                self.features_dict[cls][patient] = []

                for cell in cells:
                    if cell.lower().endswith('.h5'):
                        cell_path = os.path.join(patient_path, cell)
                        with h5py.File(cell_path, 'r') as hf:
                            feature = np.array(hf['features'])  # Assume 'features' is the dataset name in the .h5 file
                            self.features_dict[cls][patient].append(feature)

        # Flatten the dictionary to a list of samples for easy access during __getitem__
        self.samples = []
        for cls, patients in self.features_dict.items():
            cls_idx = self.class_to_idx[cls]
            for patient, features in patients.items():
                if features:  # Ensure there are features to add
                    # Stack features and store along with class label
                    self.samples.append((np.stack(features), cls_idx))
        print("data_loading done!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, cls = self.samples[idx]
        # print(cls)
        features_tensor = torch.tensor(features, dtype=torch.float)
        cls_tensor = torch.tensor(cls, dtype=torch.long)  # Adjust dtype as necessary
        return features_tensor, cls_tensor


class wbc_mil(nn.Module):

    def __init__(self, class_count, multi_attention, latent_dim) -> None:
        super(wbc_mil, self).__init__()

        
        self.attention_latent_dim = 128
        self.class_count = class_count
        self.multi_attention = multi_attention
        # self.latent_dim = latent_dim
        # self.mil_latent_dim = 500
        self.mil_latent_dim = latent_dim

        # # MIL encoder
        # self.mil_encoder = nn.Sequential(
        #     nn.Conv1d(in_channels=self.latent_dim, out_channels=int(self.latent_dim * 1.5), kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=int(self.latent_dim * 1.5), out_channels=int(self.latent_dim * 2), kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveMaxPool1d(output_size=1),
        #     nn.Flatten(),
        #     nn.Linear(int(self.latent_dim * 2), self.mil_latent_dim),
        #     nn.ReLU(),
        # )
        
        # single attention network
        self.attention = nn.Sequential(
            nn.Linear(self.mil_latent_dim, self.attention_latent_dim),
            nn.Tanh(),
            nn.Linear(self.attention_latent_dim, 1)
        )

        # multi attention network
        self.attention_multi_column = nn.Sequential(
            nn.Linear(self.mil_latent_dim, self.attention_latent_dim),
            nn.Tanh(),
            nn.Linear(self.attention_latent_dim, self.class_count),
        )

        # single head classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.mil_latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.class_count)
        )

        # multi head classifier
        self.classifier_multi_column = nn.ModuleList()
        for a in range(self.class_count):
            self.classifier_multi_column.append(nn.Sequential(
                nn.Linear(self.mil_latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ))

    def forward(self, x):

        prediction = []
        bag_feature_stack = []

        # features = self.mil_encoder(x.squeeze().unsqueeze(dim=2))
        features = x
        attention = torch.transpose(self.attention_multi_column(features), 1, 0)
        
        if self.multi_attention:
            for cls in range(self.class_count):
                # multi head aggregation
                att_softmax = F.softmax(attention.T[cls,...]) # F.softmax(attention[cls,...])
                bag_features = torch.mm(att_softmax, features.squeeze()) # torch.mm(att_softmax.unsqueeze(dim=0), features.squeeze())
                bag_feature_stack.append(bag_features)
                
                # classification
                pred = self.classifier_multi_column[cls](bag_features)
                prediction.append(pred)
                    
            prediction = torch.stack(prediction).view(1, self.class_count)
            bag_feature_stack = torch.stack(bag_feature_stack).squeeze()

            return {"prediction": prediction, "attention": attention, "att_softmax": att_softmax, "bag_features": bag_features}
        
        else:
            # single head aggregation
            att_softmax = F.softmax(attention, dim=1)
            bag_features = torch.mm(att_softmax, features)
            
            # classification
            prediction = self.classifier(bag_features)
            
            return {"prediction": prediction, "attention": attention, "att_softmax": att_softmax, "bag_features": bag_features}


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
                            Default: 5
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


class run_wbc_mil:
    def __init__(self, 
                 device, 
                 num_epochs,
                 dataset_train_val, dataset_test, batch_size,
                 class_count, multi_attention,latent_dim):
        
        self.dataset_train_val = dataset_train_val
        self.dataset_test = dataset_test
        self.batch_size = batch_size
        self.latent_dim=latent_dim

        self.class_count = class_count
        self.multi_attention = multi_attention

        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.num_epochs = num_epochs

        self.kfold_train_val = KFold(n_splits=4, shuffle=True)

        self.test_loader = DataLoader(dataset_test, batch_size=1)

    def forward(self):

        test_accuracies = []
        test_f1s = []

        for fold, (train_idx, val_idx) in enumerate(self.kfold_train_val.split(self.dataset_train_val)):
            print(f"Fold {fold + 1}")
            train_subs = Subset(self.dataset_train_val, train_idx)
            val_subs = Subset(self.dataset_train_val, val_idx)
            
            train_loader = DataLoader(train_subs, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_subs, batch_size=1)

            input_ex, _ = dataset_train_val[0]
            latent_dim = input_ex.shape[1]

            self.model = wbc_mil(class_count=self.class_count, multi_attention=self.multi_attention, latent_dim=self.latent_dim)

            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
            
            best_model = self.train_fold(fold, train_loader, val_loader)

            # Test the best model from this fold
            self.model.load_state_dict(best_model)
            self.model.eval()

            _, test_accuracy, test_f1 = self.run_epoch(self.test_loader, phase="test", batch_size=1, epoch=1)
            test_accuracies.append(test_accuracy)
            test_f1s.append(test_f1)
            print(f"Fold {fold + 1} Test Accuracy: {test_accuracy:.4f} Test F1: {test_f1:.4f}")

        return best_model, test_accuracies, test_f1s

    def train_fold(self, fold, train_loader, val_loader):
        self.model.to(self.device)
        best_loss = float('inf')

        early_stopping = EarlyStopping(patience=20, min_delta=0.0)

        for epoch in range(self.num_epochs):

            train_loss, train_accuracy, train_f1 = self.run_epoch(loader=train_loader, phase="train", batch_size=self.batch_size, epoch=epoch)
            print("fold ", fold, "epoch ", epoch, " > train loss: ", train_loss, ", train accuracy: ", train_accuracy)
            val_loss, val_accuracy, val_f1 = self.run_epoch(loader=val_loader, phase="val", batch_size=1, epoch=epoch)
            print("fold ", fold, "epoch ", epoch, " > val loss: ", val_loss, ", val accuracy: ", val_accuracy, "val_f1", val_f1)

            if val_loss < best_loss:
                best_loss = val_loss    
                best_model = self.model.state_dict()

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        return best_model

    def run_epoch(self, loader, phase, epoch, batch_size=1):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for counter, (features, cls) in enumerate(loader):
            features, labels = features.to(device), cls.to(device)

            if phase=="train":
                self.optimizer.zero_grad()

            outputs = self.model(features)
            loss = self.criterion(outputs["prediction"], labels)
            
            if phase=="train":
                loss.backward()
            total_loss += loss.item()

            if phase=="train" and ((counter+1)%batch_size==0 or (counter+1)==len(loader)):
                self.optimizer.step()
                self.scheduler.step()

            # wandb.log({f"{phase}_loss": loss.item(), "epoch": epoch})

            _, preds = torch.max(outputs["prediction"], 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate metrics
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            _, _, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
        avg_loss = total_loss / len(loader)

        # Log epoch metrics
        # wandb.log({
        #     f"{phase}_avg_loss": total_loss / len(loader), 
        #     f"{phase}_balanced_accuracy": balanced_acc, 
        #     f"{phase}_f1_weighted": f1_weighted, 
        #     "epoch": epoch
        # })

        return avg_loss, balanced_acc, f1_weighted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a WBC MIL model.')
    parser.add_argument('--data_path', 
                        type=str, 
                        required=True, 
                        help='Path to the training data directory',
                        default='dinov2_vits_orig')
    parser.add_argument('--wandb_name', 
                        type=str, 
                        help='Name of the run in wandb',
                        default='wbc_mil_dinov2_vits14')

    args = parser.parse_args()

    # wandb.init(project="histo-collab", entity="s-kazeminia-90", name=args.wandb_name) 
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    num_epochs = 150

    class_count = 5
    multi_attention = True
    batch_size = 1

    # wandb.config = {
    #     "learning_rate": 0.001,
    #     "epochs": num_epochs,
    #     "batch_size": batch_size,
    #     "class_count": class_count,
    #     "multi_attention": multi_attention,
    #     }

    root_features_path = '/lustre/groups/labs/marr/qscd01/datasets/210526_mll_mil_pseudonymized/splitted_extracted_features/'
    data_path_train = os.path.join(root_features_path, args.data_path, 'test')
    data_path_test = os.path.join(root_features_path, args.data_path, 'test')

    # Initialize dataset and dataloader
    dataset_train_val = WbcMilFeatureDataset(data_path_train)
    datset_test = WbcMilFeatureDataset(data_path_test)

    input_ex, _ = dataset_train_val[0]
    latent_dim = input_ex.shape[1]
    
    accuracies = []
    f1s = []

    for run in range(5):
        obj = run_wbc_mil(device,
                    num_epochs,
                    dataset_train_val, datset_test, batch_size,
                    class_count, multi_attention, latent_dim)
        model, acc, f1 = obj.forward()
        accuracies.append(acc)
        f1s.append(f1)
    print("Accuracy_average: ", np.mean(accuracies), " std: ", np.std(accuracies))
    print("f1_average: ", np.mean(f1s), " std: ", np.std(f1s))

    # wandb.finish()


    
