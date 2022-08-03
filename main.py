from json import load
from Models import basic
from Dataloader import Loader 

def main() :
  # Load in the data 
  loader = Loader("./intent-detection-by-sinch/X_train.npy", "./intent-detection-by-sinch/y_train.csv")
  x_data = loader.read_x_train()
  y_data = loader.read_y_train()

  # Read in the data 

if __name__ == "__main__":
    main()
