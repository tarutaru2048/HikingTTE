import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.append(project_root)
from src.common import utils
import numpy as np

from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, attribute_names):
        super(Net, self).__init__()
        self.attribute_names = attribute_names
    
    def out_size(self):
        return len(self.attribute_names)

    def forward(self, attr):
        attr_list = []
        for name in self.attribute_names:
            attr_list.append(attr[name].view(-1, 1))
        return torch.cat(attr_list, dim=1)