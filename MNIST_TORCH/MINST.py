import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np
import ssl
import useful_functions

#disable ssl to allow unverfied downloads due to 403 error from downloading the mnist datasets
ssl._create_default_https_context = ssl._create_unverified_context

#download the train and validation datasets and transfrom them to tensors 
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

#verfying that data was correctly downloaded 
print("Type of data element: ", type(train_dataset[0][1]))
print("The label: ", train_dataset[3][1])
print("\n")

#Display images to again verify that they were correctly downloaded 
print("Image from train dataset")
useful_functions.show_data(train_dataset[3])
print("\n")
print("Image from validation dataset")
useful_functions.show_data(validation_dataset[1])
