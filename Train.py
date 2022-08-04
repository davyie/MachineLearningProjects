import torch 
from Models import basic
import random as r 
from Dataloader import write_to_file

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_basic(input_size, hidden, output, x_data, y_data, number_of_datapoints, x_test_data):

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

      y_pred = model_basic(x_batch.to(device))

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
  predictions = torch.argmax(torch.softmax(model_basic(x_data.to(device)), dim=1), dim=1)

  correct_labelled = 0 
  for i in range(number_of_datapoints):
    if predictions[i] == y_data[i]:
      correct_labelled = correct_labelled + 1
  print('Correct labelled %5d out of %5d' % (correct_labelled, number_of_datapoints))
  print('Ratio %5f' % (float(correct_labelled) / float(number_of_datapoints)))


  # Predict the test data 
  test_prediction = torch.argmax(torch.softmax(model_basic(x_test_data), dim=1), dim=1)
  # print(test_prediction)
  write_to_file("y_basic_test.csv", test_prediction)

def train_deep_basic(input_size, hidden, output, x_data, y_data, number_of_datapoints, x_test_data):
  print("HELLO")