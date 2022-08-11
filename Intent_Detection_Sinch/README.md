# About this project 
This repo is about creating models for Kaggle competition of Sinch where we are suppose to predict multi class intention for different sentences. We get a sentence embedding, a vector that represents the sentence and we are suppose to output a class between 1 and 45. We do this for 1663 embeddings and we want to try and achieve higher F1 score than 0,62.

# Learning objectives 
- Work with PyTorch 
- Apply deep learning to real problem 
- Explore deep learning architectures 

# Data Loader 
This class takes in file names and returns data in terms of Tensors. These tensors are the building blocks for our network. 

# Pytorch fundamentals 
To modularise pytorch we can create a set of files with specific functionalities in mind. 
- engine.py, a file containing various training functions 
- data_setup.py, a file to prepare and download data if necessary 
- model_builder.py, a file to create pytorch models 
- train.py, a file to leverage all the other files and train a target pytorch model 
- utils.py, a file dedicated to helpful utility functions 

This is the usual structure of deep learning projects. 
# Basic solution 
This solution contains a simple neural network with an input layer of the size of embedding and reduces it to 128 dimesions. Then it computes the output probabilities of for the 45 classes. We use ReLU for the input layer and linear transformation for the output. 

The loss function is CrossEntropyLoss and we are using Stochastic Gradient Descent as optimiser. We train for 30 epochs and each epoch contains 3 minibatches of size 256. 

This solution gives us a performance of 0,62 which is the standard according to their leaderboard. 

This solution shows that simple model can predict intention class to some degree. We have been able to create a model with simple linear layers and applied relu activation function between for input layer and then applied softmax to the output layer to obtain probabilities. 

# Deep basic solution 
This solution consist of a three layer network with one input layer of size 1663x128, hidden layer of size 128x128 and output layer of size 1663x45. The activation for hidden layer is ReLU and the output will be softmax. 
We try different configurations with the number of neurons at the hidden layers. 

We try 128x128, 256x128 and 512x256. 
The 128x128 performs very bad and it doesn't learn anything. 256x128 learns a bit more and 512x256 gets the best correctly labelled datapoints for training data. With further exploration with minibatch we are able to achieve a good performance with 128x128 where the engine goes through the data sequentially rather than random. 

They are trained for 30 epochs with a batchsize of 256 and the number of batches is 1000. 

# Feedforward network with dropout matrix 
This solution contains a feedforward network with dropout matrix of the input where we replace certain embeddings. Additionally, we add `tanh` as activation function instead of Rectified Linear Unit. This gives us a performance of 0.67 which is above 0.62. This is the best one so far. 