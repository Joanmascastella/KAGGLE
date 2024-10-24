import matplotlib.pylab as plt
import torch
import torch.nn as nn


# The function to plot parameters
def PlotParameters(model): 
    W = model.state_dict()['fc1.weight'].data
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.01, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        if i < 10:
            
            # Set the label for the sub-plot.
            ax.set_xlabel("class: {0}".format(i))

            # Plot the image.
            ax.imshow(W[i, :].view(28, 28), vmin=w_min, vmax=w_max, cmap='seismic')

            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
    plt.show()

# The function that displays image from dataset
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))
    plt.show()

# Plot the loss and accuracy
def plot_loss_accuracy(loss_list, accuracy_list):
    fig, ax1 = plt.subplots()

    # Plotting the loss on the first y-axis
    color = 'tab:red'
    ax1.plot(loss_list, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Creating a second y-axis for the accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)
    ax2.plot(accuracy_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Adjust layout to prevent overlapping labels
    fig.tight_layout()

    # Show the plot
    plt.show()

# Plot the misclassified samples
def display_misclassified(validation_dataset, model):

    Softmax_fn=nn.Softmax(dim=-1)
    count = 0
    for x, y in validation_dataset:
        z = model(x.reshape(-1, 28 * 28))
        _, yhat = torch.max(z, 1)
        if yhat != y:
            show_data((x, y))
            plt.show()
            print("yhat:", yhat)
            print("probability of class ", torch.max(Softmax_fn(z)).item())
            count += 1
        if count >= 5:
            break 

# Plot the correctly classified samples
def display_correct(validation_dataset, model):
    Softmax_fn=nn.Softmax(dim=-1)
    count = 0
    for x, y in validation_dataset:
        z = model(x.reshape(-1, 28 * 28))
        _, yhat = torch.max(z, 1)
        if yhat == y:
            show_data((x, y))
            plt.show()
            print("yhat:", yhat)
            print("probability of class ", torch.max(Softmax_fn(z)).item())
            count += 1
        if count >= 5:
            break  