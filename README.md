# About this repo 
This repo is about creating models for Kaggle competition of Sinch where we are suppose to predict multi class intention for different sentences. We get a sentence embedding, a vector that represents the sentence and we are suppose to output a class between 1 and 45. We do this for 1663 embeddings and we want to try and achieve higher F1 score than 0,62.

# Learning objectives 
- Work with PyTorch 
- Apply deep learning to real problem 
- Explore deep learning architectures 

# Data Loader 
This class takes in file names and returns data in terms of Tensors. These tensors are the building blocks for our network. 

# Basic solution 
This solution contains a simple neural network with an input layer of the size of embedding and reduces it to 128 dimesions. Then it computes the output probabilities of for the 45 classes. We use ReLU for the input layer and linear transformation for the output. 

The loss function is CrossEntropyLoss and we are using Stochastic Gradient Descent as optimiser. We train for 30 epochs and each epoch contains 3 minibatches of size 256. 

This solution gives us a performance of 0,62 which is the standard according to their leaderboard. 

This solution shows that simple model can predict intention class to some degree. We have been able to create a model with simple linear layers and applied relu activation function between for input layer and then applied softmax to the output layer to obtain probabilities. 

# Deep basic solution 

