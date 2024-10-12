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

#Creating softmax classfier 
class SoftMaxClassifier(nn.Module):
    # Constructor
    def __init__(self, input_size, hidden_size, output_size):
        super(SoftMaxClassifier, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer (input to hidden)
        self.relu = nn.ReLU()                         # Activation function (ReLU)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer (hidden to output)
        
    # Forward pass
    def forward(self, x):
        x = self.fc1(x)    # Pass input through the first layer
        x = self.relu(x)   # Apply ReLU activation
        x = self.fc2(x)    # Pass through the second layer
        return x
    
#definng the input and output size of out nueral network 
input_dim = 28 * 28
hidden_dim = 64
output_dim = 10

#viewing the initial weights and baises visualized to make sure nn is correctly initiliazed 
model = SoftMaxClassifier(input_dim, hidden_dim, output_dim)
useful_functions.PlotParameters(model)

# Define the learning rate, optimizer, criterion and data loader
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

#Train model 
n_epochs = 10
loss_list = []
accuracy_list = []
N_test = len(validation_dataset)

def train_model(n_epochs):
    for epoch in range(n_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            
        correct = 0
        # perform a prediction on the validation data  
        for x_test, y_test in validation_loader:
            z = model(x_test.view(-1, 28 * 28))
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        loss_list.append(loss.data)
        accuracy_list.append(accuracy)

train_model(n_epochs)

# Call the function to plot loss and accuracy
useful_functions.plot_loss_accuracy(loss_list, accuracy_list)

# Call the function to display misclassified items 
useful_functions.display_misclassified(validation_dataset, model)

#Call the function to display the correctly classified items
useful_functions.display_correct(validation_dataset, model)