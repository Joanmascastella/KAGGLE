import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import numpy as np
import ssl
import useful_functions

# Disable SSL verification to allow unverified downloads due to 403 errors from downloading the MNIST datasets
ssl._create_default_https_context = ssl._create_unverified_context

# Data Augmentation: Add random rotation, horizontal flip, and normalization to the training set
train_transforms = transforms.Compose([
    transforms.RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean 0.5 and std 0.5
])

# Validation set (no augmentation)
validation_transforms = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean 0.5 and std 0.5
])

# Download the train and validation datasets with the respective transforms
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=train_transforms)
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=validation_transforms)

# Verify that the data was correctly downloaded
print("Type of data element: ", type(train_dataset[0][1]))
print("The label: ", train_dataset[3][1])

# Display images to verify augmentation (will show an augmented version from the training set)
print("Image from train dataset (with augmentation)")
useful_functions.show_data(train_dataset[3])
print("\n")
print("Image from validation dataset (no augmentation)")
useful_functions.show_data(validation_dataset[1])

# Creating a SoftMax classifier model
class SoftMaxClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SoftMaxClassifier, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden to output layer
        
    def forward(self, x):
        x = self.fc1(x)    # First layer
        x = self.relu(x)   # Activation
        x = self.fc2(x)    # Second layer
        return x

# Define the input, hidden, and output sizes
input_dim = 28 * 28
hidden_dim = 64
output_dim = 10

# Create the model and visualize the initial weights
model = SoftMaxClassifier(input_dim, hidden_dim, output_dim)
useful_functions.PlotParameters(model)

# Define the learning rate, optimizer, and loss function
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Create data loaders for train and validation datasets
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Train model
n_epochs = 10
loss_list = []
accuracy_list = []
N_test = len(validation_dataset)

def train_model(n_epochs):
    for epoch in range(n_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))  # Flatten the image
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

        correct = 0
        # Perform validation
        for x_test, y_test in validation_loader:
            z = model(x_test.view(-1, 28 * 28))
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        loss_list.append(loss.item())
        accuracy_list.append(accuracy)

train_model(n_epochs)

# Print final accuracy
final_accuracy = accuracy_list[-1]  # Get the last recorded accuracy
print(f"Final validation accuracy after {n_epochs} epochs: {final_accuracy * 100:.2f}%")

# Call the function to plot loss and accuracy
useful_functions.plot_loss_accuracy(loss_list, accuracy_list)

# Call the function to display misclassified items 
useful_functions.display_misclassified(validation_dataset, model)

# Call the function to display the correctly classified items
useful_functions.display_correct(validation_dataset, model)

useful_functions.PlotParameters(model)
