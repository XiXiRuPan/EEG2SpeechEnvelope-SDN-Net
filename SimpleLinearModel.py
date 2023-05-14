import os
import torch
import torch.nn as nn
import pdb
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

import json
class SimpleLinearModel(nn.Module):
    def __init__(self, integration_window=1, nb_filters=1, nb_channels=64):
        super(SimpleLinearModel, self).__init__()
        #class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1d = nn.Conv1d(nb_channels, nb_filters, integration_window)

    def forward(self, x):
        x = self.conv1d(x)
        return x
