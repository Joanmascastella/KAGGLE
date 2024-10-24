import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Helper function to map computation to MPS (or CUDA/CPU)
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple M1/M2/M3 GPUs
        print("Using MPS device for computations.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
        print("Using CUDA device for computations.")
    else:
        device = torch.device("cpu")  # Fallback to CPU
        print("Using CPU device for computations.")
    return device


# Helper function to plot the training loss
def plot_metrics_with_dual_axes(loss_list, mae_list=None, title="Training Loss and Accuracy Over Epochs"):
    """
    Plots loss and accuracy (MAE) over epochs, using a secondary axis for MAE if needed.

    :param loss_list: List of loss values collected during training.
    :param mae_list: List of MAE values collected during training.
    :param title: Title for the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot loss on the primary y-axis
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(loss_list, label="Loss", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot MAE on the secondary y-axis
    if mae_list is not None:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('MAE (Accuracy)', color='tab:orange')  # we already handled the x-label with ax1
        ax2.plot(mae_list, label="MAE (Accuracy)", color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title(title)
    fig.tight_layout()  # to prevent overlap of y-axis labels
    plt.grid(True)
    plt.show()

# Helper function to plot the training loss
def autoencoder_plot_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# From GPT
# Custom RMSE criterion with log transformation
def rmse_log_loss(y_pred, y):
    # Clamp values to avoid taking log of zero or negative values
    y_pred = torch.clamp(y_pred, min=1e-6)
    y = torch.clamp(y, min=1e-6)

    # Take the log of predictions and true values
    y_pred_log = torch.log(y_pred)
    y_log = torch.log(y)

    # Calculate Mean Squared Error between log values
    mse_loss = F.mse_loss(y_pred_log, y_log)

    # Return Root Mean Squared Error (RMSE)
    return torch.sqrt(mse_loss)