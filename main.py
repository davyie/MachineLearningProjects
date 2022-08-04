from Dataloader import Loader 
from Train import train_basic

def main() :
  # Load in the data 
  loader = Loader("./intent-detection-by-sinch/X_train.npy", "./intent-detection-by-sinch/y_train.csv")
  x_data = loader.read_x_train()
  y_data = loader.read_y_train()
  x_test_data = loader.read_x_test("./intent-detection-by-sinch/X_test.npy")

  # Model
  number_of_datapoints, input_size = loader.get_x_shape()
  hidden = 128
  output = 45

  train_basic(input_size=input_size, hidden=hidden, output=output, x_data=x_data, y_data=y_data, number_of_datapoints=number_of_datapoints, x_test_data=x_test_data)
  

if __name__ == "__main__":
    main()
