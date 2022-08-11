from turtle import forward
import torch 
import torch.nn as nn

# Inherit the nn.Module 
class basic(torch.nn.Module):


  def __init__(self, input_size, hidden_size, output_size):
    super(basic, self).__init__()
    self.layer_1 = nn.Linear(input_size, hidden_size)
    self.output = nn.Linear(hidden_size, output_size)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()

  def forward(self, x):
    # The forward pass of the model 
    ins = self.relu(self.layer_1(x))
    output = self.output(ins)
    return output

class deep_basic(torch.nn.Module):

  def __init__(self, input_size, hidden_size, output_size) -> None:
    super(deep_basic, self).__init__()
    self.input_layer = nn.Linear(input_size, hidden_size)
    self.hidden_layer = nn.Linear(hidden_size, hidden_size)
    self.output_layer = nn.Linear(hidden_size, output_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    ins = self.relu(self.input_layer(x))
    hidden_ins = self.relu(self.hidden_layer(ins))
    output = self.output_layer(hidden_ins)
    return output
