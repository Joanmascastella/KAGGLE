import torch
import torch.nn as nn
import torch.optim as optim
import helpful_functions as hf
from sklearn.model_selection import train_test_split
import config


# Define the fully connected network for house price prediction
class HousePriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousePriceModel, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 128)  # First layer (input to hidden)
        self.relu1 = nn.ReLU()  # Activation function

        self.fc2 = nn.Linear(128, 64)  # Second hidden layer
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 32)  # Third hidden layer
        self.relu3 = nn.ReLU()

        self.output = nn.Linear(32, 1)  # Output layer (single value for house price prediction)

        # Initialize weights using Kaiming (He) initialization for better convergence
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.output.weight, nonlinearity="relu")

    def forward(self, x):
        # Pass through the fully connected layers with ReLU activations
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.output(x)  # Output is a single value (house price)
        return x

device = hf.get_device()

def define_parameters(model):
    model = model

    # Define the optimizer, learning rate, and loss function
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train model
    n_epochs = 100
    loss_list = []
    accuracy_list = []

    return model, optimizer, criterion, loss_list, accuracy_list, n_epochs

def compile_and_train_model(train_features, test_features, model, optimizer, criterion, loss_list, accuracy_list, n_epochs):
    print(train_features)  # Inspect the structure
    def train(n_epochs):
        for epoch in range(n_epochs):
            for x, y in train_features:
                x, y = x.to(device), y.to(device)
                z = model(x)
                loss = criterion(z, y)
                loss.backward()
                optimizer.step()

        correct = 0
        #Perfom Validation
        for x, y in test_features:
            x, y = x.to(device), y.to(device)
            z = model(x)
            _, predicted = torch.max(z.data, 1)
            correct += (predicted == y).sum().item()
        accuracy = correct / len(test_features)
        loss_list.append(loss.item())
        accuracy_list.append(accuracy)

    train(n_epochs)
    return loss_list, accuracy_list