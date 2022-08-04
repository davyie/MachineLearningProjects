from Models import basic
from Dataloader import Loader 

import torch 
import random as r
import numpy as np

def main() :
  # Load in the data 
  loader = Loader("./intent-detection-by-sinch/X_train.npy", "./intent-detection-by-sinch/y_train.csv")
  x_data = loader.read_x_train()
  y_data = loader.read_y_train()

  # Model
  number_of_datapoints, input_size = loader.get_x_shape()
  hidden = 128
  output = 45

  # Training 
  epochs = 30
  batch_size = 256
  number_of_batches = 2048
  model_basic = basic(input_size, hidden, output)
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model_basic.parameters(), lr=1e-6)

  current_loss = 0.0
  for t in range(epochs):
    for batch_id in range(number_of_batches):
      start = r.randint(0, number_of_datapoints - batch_size)

      x_batch = x_data[start:start+batch_size, :]
      y_batch = y_data[start:start+batch_size]

      y_pred = model_basic(x_batch)

      loss = criterion(y_pred, y_batch)

      loss.backward()
      optimizer.step()
      # Print statistics
      current_loss += loss.item()
      if batch_id % 500 == 499:
        print('Loss after epoch %5d and mini-batch %5d: %.3f' %
              (t, batch_id + 1, current_loss / 500))
        current_loss = 0.0

  # Predict the training data 
  print("Prediction")
  predictions = torch.argmax(model_basic(x_data), dim=1)

  correct_labelled = 0 
  for i in range(number_of_datapoints):
    if predictions[i] == y_data[i]:
      correct_labelled = correct_labelled + 1
  print('Correct labelled %5d out of %5d' % (correct_labelled, number_of_datapoints))
  print('Ratio %5f' % (float(correct_labelled) / float(number_of_datapoints)))


  # Predict the test data 
  x_test_data = loader.read_x_test('./intent-detection-by-sinch/X_test.npy')
  test_prediction = torch.argmax(model_basic(x_test_data), dim=1)
  # print(test_prediction)
  write_to_file(test_prediction)

def write_to_file(data):
  f = open("y_test.csv", "a")
  f.write("Id,Predicted\n")
  for k, v in enumerate(data):
    f.write('%d,%d\n' % (k, v))
  

if __name__ == "__main__":
    main()
