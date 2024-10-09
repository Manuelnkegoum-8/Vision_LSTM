from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2  # Using the newer v2 transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# Define CIFAR10Policy transformations

def create_mnist_datasets(height,width):
    # Training transforms
    train_transform = v2.Compose([
        v2.Resize((height,width)),  # Resize to slightly larger size for RandomResizedCrop
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        #v2.Normalize() 
    ])
    # Test transforms
    test_transform = v2.Compose([
        v2.Resize((height,width)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        #v2.Normalize()
    ])
    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


def create_cifar_datasets(height,width):
    # Training transforms
    train_transform = v2.Compose([
        v2.Resize((height,width)),  # Resize to slightly larger size for RandomResizedCrop
        v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        #v2.Normalize() 
    ])
    # Test transforms
    test_transform = v2.Compose([
        v2.Resize((height,width)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        #v2.Normalize()
    ])
    # Download and load the training data
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    # Download and load the test data
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


class ConfigVil:
    def __init__(self,n_classes, m_layers, dim,mlp_dim, qk_size, dropout_rate,height,width,patch_size,channels,classif):
        self.n_classes = n_classes          # Vocabulary size     # Embedding dimension
        self.m_layers = m_layers               # Number of LSTM layers
        self.dim = dim                         # Hidden dimension size
        self.mlp_dim = mlp_dim
        self.qk_size = qk_size                  # Number of heads (for multi-head attention if needed)
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.channels = channels
        self.dropout_rate = dropout_rate       # Dropout rate for regularization
        self.classif = classif

def calculate_accuracy(predictions, labels):
    """
    Calculate the accuracy of predictions compared to true labels.
    
    Args:
    predictions (torch.Tensor): Model predictions. Shape: (batch_size, num_classes)
    labels (torch.Tensor): True labels. Shape: (batch_size,)
    
    Returns:
    float: Accuracy as a percentage
    """
    # Get the predicted class (highest probability)
    _, predicted_classes = torch.max(predictions, dim=1)
    
    # Compare predictions with true labels
    correct_predictions = (predicted_classes == labels).float()
    
    # Calculate accuracy
    accuracy = correct_predictions.sum() / len(correct_predictions)
    
    return accuracy.item() * 100  # Convert to percentage
    
@torch.no_grad()
def generate(model,images):
    pass
    #return preds