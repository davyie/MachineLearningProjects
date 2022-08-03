import numpy as np 
import pandas as pd
import torch 

class Loader:
  
  def __init__(self, x_train_path, y_train_path):
    self.x_train_path = x_train_path
    self.y_train_path = y_train_path

  def get_x_path(self):
    return self.x_train_path

  def get_y_path(self):
    return self.y_train_path

  def read_x_train(self):
    self.x_train = np.load(self.x_train_path)
    # Has shape (1663, 768)
    # Transform to tensor 
    return torch.tensor(self.x_train)

  def read_y_train(self):
    self.y_train = pd.read_csv(self.y_train_path).Predicted.to_numpy()
    # Has shape (1663, )
    return torch.tensor(self.y_train)

  def get_x_shape(self):
    return self.x_train.shape

  def get_y_shape(self):
    return self.y_train_shape

  def read_x_test(self, path):
    data = np.load(path)
    return torch.tensor(data)

  
  