from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aggregators.aggregator import BaseAggregator


"""

Implement Attention MIL for the unimodal (WSI only) and multimodal setting (pathways + WSI). The combining of modalities 
can be done using bilinear fusion or concatenation. 

Mobadersany, Pooya, et al. "Predicting cancer outcomes from histology and genomics using convolutional networks." Proceedings of the National Academy of Sciences 115.13 (2018): E2970-E2979.

"""

################################
# Attention MIL Implementation #
################################
class ABMIL(BaseAggregator):
    def __init__(self, input_dim=None, size_arg = "big", n_heads=1, dropout=0.25, num_classes=4):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            n_heads (int): Number of parallel attention branches / networks
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(BaseAggregator, self).__init__()
        self.size_dict = {"small": [input_dim, 256, 256], "big": [input_dim, 512, 384]}

        ### Deep Sets Architecture Construction
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.input_dim = input_dim
        self.classifier = nn.Linear(size[2]*n_heads, num_classes)

        self.activation = nn.ReLU()

    def forward(self, x, *args):
        x_path = x
        x_path = x_path.squeeze(0) #---> need to do this to make it work with this set up
        A, h_path = self.attention_net(x_path)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()
        
        h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        
        return logits


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x