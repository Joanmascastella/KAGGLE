import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pylab as plt
import ssl
import useful_functions

# Disable SSL verification to allow unverified downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Data Augmentation for training set
train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Validation set transforms (no augmentation)
validation_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download the MNIST datasets
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=train_transforms)
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=validation_transforms)

# Creating a Convolutional Neural Network (CNN) Model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # First convolutional layer (input: 1 channel, output: 16 feature maps, kernel size: 3x3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()  # ReLU activation
        
        # Second convolutional layer (input: 16 channels, output: 32 feature maps, kernel size: 3x3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()  # ReLU activation
        
        # Max pooling layer (2x2 pool size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional layer (input: 32 channels, output: 64 feature maps, kernel size: 3x3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()  # ReLU activation

        # Fully connected layer (flattened 7x7x64 input, 128 output neurons)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu_fc = nn.ReLU()

        # Output layer (128 neurons to 10 output classes)
        self.fc2 = nn.Linear(128, 10)
        
        # Initialize weights with Kaiming normalization
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        # Pass through the convolutional layers with ReLU and pooling
        x = self.pool(self.relu1(self.conv1(x)))  # conv1 -> relu -> pool
        x = self.pool(self.relu2(self.conv2(x)))  # conv2 -> relu -> pool
        x = self.relu3(self.conv3(x))  # conv3 -> relu (no pooling)
        
        # Flatten the feature maps
        x = x.view(-1, 64 * 7 * 7)
        
        # Pass through fully connected layers
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x

# Define the model
model = CNNClassifier()
# useful_functions.PlotParameters(model)

# Define the optimizer, learning rate, and loss function
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Create data loaders for training and validation
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
            z = model(x)  # Input is now passed as is (not flattened)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

        correct = 0
        # Perform validation
        for x_test, y_test in validation_loader:
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        loss_list.append(loss.item())
        accuracy_list.append(accuracy)

train_model(n_epochs)

# Print final accuracy
final_accuracy = accuracy_list[-1]
print(f"Final validation accuracy after {n_epochs} epochs: {final_accuracy * 100:.2f}%")

# # Call the function to plot loss and accuracy
# useful_functions.plot_loss_accuracy(loss_list, accuracy_list)

# # Call the function to display misclassified items 
# useful_functions.display_misclassified(validation_dataset, model)

# # Call the function to display the correctly classified items
# useful_functions.display_correct(validation_dataset, model)

# Plot the model parameters (optional)
# useful_functions.PlotParameters(model)