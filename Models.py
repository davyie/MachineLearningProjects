from turtle import forward
import torch 
import torch.nn as nn

# Inherit the nn.Module 
class basic(torch.nn.Module):


  def __init__(self, input_size, hidden_size, output_size, bias=True):
    super(basic, self).__init__()
    self.layer_1 = nn.Linear(input_size, hidden_size)
    # self.layer_2 = nn.Linear(hidden_size, hidden_size)
    self.output = nn.Linear(hidden_size, output_size)
    self.relu = nn.ReLU()


  def forward(self, x):
    # The forward pass of the model 
    ins = self.relu(self.layer_1(x))
    # ins = self.relu(self.layer_2(ins))
    output = self.output(ins)
    return output