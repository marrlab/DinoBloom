import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from models.dinov2 import vit_small

DINO_VIT_S_PATH = '/home/aih/benedikt.roth/dino_train_res/eval/training_18499/teacher_checkpoint.pth'

def get_dino_vit_s():
    vit_kwargs = dict(
        img_size=224,
        patch_size=14,
        #init_values=1.0,
        #ffn_layer="mlp",
        #block_chunks=0,
    )
    model = vit_small(**vit_kwargs)
    pretrained = torch.load(DINO_VIT_S_PATH)
    new_state_dict = {}

    for key, value in pretrained['teacher'].items():
        new_key = key.replace('backbone.', '')
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    return model

class MyModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = get_dino_vit_s()
        self.linear_layer = nn.Linear(in_features=384, out_features=num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        output = self.linear_layer(features)
        return output
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == targets).float().mean()
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.linear_layer.parameters(), lr=0.001)
        return optimizer