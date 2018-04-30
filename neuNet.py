import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class net1(nn.Module):
  def __init__(self):
    super(net1,self).__init__()
    self.fc1 = nn.Linear(9,50)
    self.fc2 = nn.Linear(50,30)
    self.fc3 = nn.Linear(30,10)
    self.fc4 = nn.Linear(10,2)
    
  def forward(self,x):
    x1 = F.tanh(self.fc1(x))
    x2 = F.tanh(self.fc2(x1))
    x3 = F.tanh(self.fc3(x2))
    x4 = self.fc4(x3)
    return x4
  


class net2(nn.Module):
  def __init__(self):
    super(net2,self).__init__()
    self.fc1 = nn.Linear(9,70)
    self.fc2 = nn.Linear(70,50)
    self.fc3 = nn.Linear(50,30)
    self.fc4 = nn.Linear(30,15)
    self.fc5 = nn.Linear(15,2)
    
  def forward(self,x):
    x1 = F.tanh(self.fc1(x))
    x2 = F.tanh(self.fc2(x1))
    x3 = F.tanh(self.fc3(x2))
    x4 = F.tanh(self.fc4(x3))
    x5 = self.fc5(x4)
    return x5