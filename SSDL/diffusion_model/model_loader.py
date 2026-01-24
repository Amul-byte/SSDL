import os
import torch
import random
import numpy as np
import torch.nn as nn
from .model import Diffusion1D
from .sensor_model import CombinedLSTMClassifier

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_weights(m):
    """
    Initialize model weights using Xavier uniform initialization.
    Args:
        m (nn.Module): Model layer or module.
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def load_sensor_model(args, device):
    """
    Load the sensor model, initialize weights, and load pre-trained model if necessary.
    Args:
        args: Arguments containing model configuration and paths.
        device (torch.device): Device to load the model on.
    Returns:
        CombinedLSTMClassifier: Loaded sensor model.
    """
    # Set seed for reproducibility
    set_seed(args.seed)

    # Initialize the sensor model
    model = CombinedLSTMClassifier(
        sensor_input_size=3, 
        hidden_size=256, 
        num_layers=8, 
        num_classes=12, 
        conv_channels=16, 
        kernel_size=3, 
        dropout=0.5, 
        num_heads=4
    ).to(device)

    # Apply weight initialization
    model.apply(initialize_weights)

    # Load pre-trained model if not in training mode
    if not args.train_sensor_model:
        sensor_model_path = os.path.join(args.output_dir, "sensor_model", "best_sensor_model.pth")
        if os.path.exists(sensor_model_path):
            checkpoint = torch.load(sensor_model_path, map_location=device)
            # Handle 'module.' prefix for models saved with DataParallel/DistributedDataParallel
            if 'module.' in next(iter(checkpoint.keys())):
                checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
            model.eval()  # Set model to evaluation mode
            print(f"Loaded pre-trained sensor model from {sensor_model_path}")
        else:
            raise FileNotFoundError(f"No pre-trained sensor model found at {sensor_model_path}")
    
    return model

def load_diffusion(device):
    """
    Load the Diffusion1D model onto the specified device.
    Args:
        device (torch.device): Device to load the model on.
    Returns:
        Diffusion1D: Loaded diffusion model.
    """
    model = Diffusion1D().to(device)
    return model

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from the keys in the state dict.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    return new_state_dict

def load_diffusion_model_for_testing(device, output_dir, test_diffusion_model):
    model = Diffusion1D().to(device)
    
    checkpoint_path = os.path.join(output_dir, "diffusion_model", "best_diffusion_model.pth")
    print(f"path is {checkpoint_path}")
    if test_diffusion_model:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # print(checkpoint.keys())
            if any(key.startswith('module.') for key in checkpoint.keys()):
                print("Removing 'module.' prefix from state_dict keys...")
                state_dict = remove_module_prefix(checkpoint)

            model.load_state_dict(state_dict)

            print(f"Loaded diffusion model from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Initializing new model.")
    else:
        print("Initializing new diffusion model without loading checkpoint.")
    
    return model