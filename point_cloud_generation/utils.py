import torch

def save_model(model, filename="point_cloud_model.pth"):
    """Saves the model state to a file."""
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model, filename="point_cloud_model.pth"):
    """Loads the model state from a file."""
    try:
        model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")
    except FileNotFoundError:
        print(f"Model file {filename} not found. Starting training from scratch.")