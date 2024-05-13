import torch
import torch.nn as nn
class HintonTower(nn.Module):

  def __init__(self, input_size, output_size, hidden_size, n_subclass, p=.4):
     super(HintonTower, self).__init__()
     self.n_subclasses = n_subclass
     self.output_size = output_size
     self.fc1 = nn.Linear(input_size, hidden_size)
     self.fc2 = nn.Linear(hidden_size,output_size*n_subclass)
     self.relu = nn.ReLU()
     self.dropout = nn.Dropout(p)
     self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
     out = self.fc1(x)
     out = self.relu(out)
     out = self.dropout(out)
     out = self.fc2(out)
     out = self.softmax(out)
     out = out.view(-1,self.output_size, self.n_subclasses)
     out = torch.sum(out, dim=2)
     return out

class Tower(nn.Module):

  def __init__(self, input_size, output_size, hidden_size, p=.4):
     super(Tower, self).__init__()
     self.fc1 = nn.Linear(input_size, hidden_size)
     self.fc2 = nn.Linear(hidden_size, output_size)
     self.relu = nn.ReLU()
     self.dropout = nn.Dropout(p)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.fc2(out)
    return out
