import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split, StratifiedShuffleSplit
import random


# from transformers import ViTModel


class wbc_feat_Dataset(Dataset):
    # def __init__(self, features, labels):
    #     self.features = features
    #     self.labels = labels
    def __init__(self, data_path):
        """
        Args:
            features (Tensor): Precomputed features for your dataset.
            labels (Tensor): Labels for your dataset.
        """
        # load treain data
        self.features, self.labels = load_extracted_features(data_path) # ToDo be defined

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

class wbc_mil(nn.Module):

    def __init__(self, class_count, multi_attention) -> None:
        super(wbc_mil, self).__init__()

        self.latent_dim = 768
        self.attention_latent_dim = 128
        self.class_count = class_count
        
        # self.feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.multi_attention = multi_attention
        
        # single attention network
        self.attention = nn.Sequential(
            nn.Linear(self.latent_dim,self.attention_latent_dim),
            nn.Tanh(),
            nn.Linear(self.attention_latent_dim, 1)
        )

        # multi attention network
        self.attention_multi_column = nn.Sequential(
            nn.Linear(self.latent_dim,self.attention_latent_dim),
            nn.Tanh(),
            nn.Linear(self.attention_latent_dim, self.class_count),
        )

        # single head classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.class_count)
        )

        # multi head classifier
        self.classifier_multi_column = nn.ModuleList()
        for a in range(self.class_count):
            self.classifier_multi_column.append(nn.Sequential(
                nn.Linear(self.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ))

    def forward(self, x):

        prediction = []
        bag_feature_stack = []

        # features = self.feature_extractor(x).last_hidden_state
        features = x
        attention = torch.transpose(self.attention_multi_column(features), 1, 0)
        
        if self.multi_attention:
            for cls in range(self.class_count):
                # multi head aggregation
                att_softmax = F.softmax(attention[..., cls], dim=1)
                bag_features = torch.mm(att_softmax.T, features.squeeze())
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


class run_wbc_mil:
    def __init__(self, 
                 device,
                 criterion, 
                 num_epochs,
                 dataset_train_val, dataset_test, batch_size,
                 model, class_count, multi_attention, optimizer):
        
        self.dataset_train_val = dataset_train_val
        self.dataset_test = dataset_test
        self.batch_size = batch_size

        self.model = model
        self.class_count = class_count
        self.multi_attention = multi_attention

        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

        self.kfold_train_val = KFold(n_splits=4, shuffle=True)

        self.test_loader = DataLoader(dataset_test, batch_size=1)

    def forward(self):

        test_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(self.kfold_train_val.split(self.dataset_train_val)):
            print(f"Fold {fold + 1}")
            train_subs = Subset(self.dataset_train_val, train_idx)
            val_subs = Subset(self.dataset_train_val, val_idx)
            
            train_loader = DataLoader(train_subs, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_subs, batch_size=1)

            self.model = wbc_mil(class_count=self.class_count, multi_attention=self.multi_attention)
            
            best_model = self.train_fold(fold, train_loader, val_loader)

            # Test the best model from this fold
            self.model.load_state_dict(best_model)
            self.model.eval()

            _, test_accuracy = self.run_epoch(self.test_loader, train=False, batch_size=1)
            test_accuracies.append(test_accuracy)
            print(f"Fold {fold + 1} Test Accuracy: {test_accuracy:.4f}")

        return best_model, test_accuracies

    def train_fold(self, fold, train_loader, val_loader):
        self.model.to(self.device)
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.model.train()

            train_loss, train_accuracy = self.run_epoch(train_loader, train=True, batch_size=self.batch_size)
            print("fold ", fold, "epoch ", epoch, " > train loss: ", train_loss, ", train accuracy: ", train_accuracy)
            val_loss, val_accuracy = self.run_epoch(val_loader, train=True, batch_size=1)
            print("fold ", fold, "epoch ", epoch, " > val loss: ", val_loss, ", val accuracy: ", val_accuracy)

            if val_loss < best_loss:
                best_loss = val_loss    
                best_model = self.model.state_dict()

        return best_model

    def run_epoch(self, loader, train, batch_size=1):
        total_loss = 0.0
        corrects = 0
        
        for counter, data in enumerate(loader):
            features, labels = data #['features'].to(self.device), loader['labels'].to(self.device)
            
            if train:
                self.optimizer.zero_grad()

            outputs = self.model(features)
            loss = self.criterion(outputs["prediction"], labels)
            loss.backward()
            total_loss += loss.item()

            if train and ((counter+1)%batch_size==0 or (counter+1)==self.dataset.__len__()):
                self.optimizer.step()

            preds = outputs["prediction"].argmax(dim=1)
            corrects += torch.sum(preds == labels).item()

        avg_loss = total_loss / len(loader)
        accuracy = corrects / len(loader.dataset)
        return avg_loss, accuracy


def example_data():
    B = 100  # Total number of bags
    D = 768  # Dimensionality of each instance's feature vector
    class_count = 5  # Number of classes
    min_instances = 5  # Minimum number of instances per bag
    max_instances = 15  # Maximum number of instances per bag

    # Initialize lists to hold the features and labels
    features = []
    labels = torch.randint(0, class_count, (B,))  # Random labels for each bag

    for _ in range(B):
        N = random.randint(min_instances, max_instances)
        bag_features = torch.randn(N, D)
        features.append(bag_features)

    return features, labels

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5

    class_count = 5
    multi_attention = True
    batch_size = 2

    data_path_train = '/lustre/groups/labs/marr/qscd01/datasets/210526_mll_mil_pseudonymized/splitted_data/train'
    data_path_test = '/lustre/groups/labs/marr/qscd01/datasets/210526_mll_mil_pseudonymized/splitted_data/test'
    # features, labels = example_data()

    # Initialize dataset and dataloader
    dataset_train_val = wbc_feat_Dataset(data_path_train)
    datset_test = wbc_feat_Dataset(data_path_test)
    # dataset = wbc_feat_Dataset(features, labels)

    # # train and test split
    # all_labels = [label for _, label in dataset]
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    # all_labels_np = np.array(all_labels)
    # train_val_idx, test_idx = next(sss.split(np.zeros(len(all_labels_np)), all_labels_np))
    # train_val_dataset = Subset(dataset, train_val_idx)
    # test_dataset = Subset(dataset, test_idx)

    # Initialize the model & optimizer
    model = wbc_mil(class_count=class_count, multi_attention=multi_attention)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    accuracies = []

    for run in range(2):
        obj = run_wbc_mil(device,
                    criterion,
                    num_epochs,
                    dataset_train_val, datset_test, batch_size,
                    model, class_count, multi_attention, optimizer)
        model, accuracy = obj.forward()
        accuracies.append(accuracy)
    print("Accuracy_average: ", np.mean(accuracies), " std: ", np.std(accuracies))

    
