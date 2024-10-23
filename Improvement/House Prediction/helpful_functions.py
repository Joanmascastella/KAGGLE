import torch
import matplotlib.pyplot as plt

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
def autoencoder_plot_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()