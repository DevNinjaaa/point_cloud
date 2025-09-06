import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os

from point_cloud_generation.model import PointCloudGenerationModel
from point_cloud_generation.dataset import CustomDataset, prepare_datasets
from point_cloud_generation.utils import save_model

def train_model():
    # Ensure the data is prepared before training
    prepare_datasets()

    # Hyperparameters
    learning_rate = 1e-4
    batch_size = 32
    epochs = 50

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Datasets and DataLoaders
    # The dataset class will now handle finding the data from the new 'data' directory
    train_ds = CustomDataset("train", img_ds_path="./data/classes", pt_cloud_ds_path="./data/ModelNet40_ply")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # You will need to find the correct links for the dataset and a corresponding image dataset.
    # The ModelNet40 dataset is available from Princeton University.

    # *If a validation dataset exists, you can uncomment this*
    # val_ds = CustomDataset("val", img_ds_path="./data/classes", pt_cloud_ds_path="./data/ModelNet40_ply")
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer
    model = PointCloudGenerationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img_input = data[0].to(device)
            pt_cloud_target = data[1].to(device)

            # Forward pass
            outputs = model(img_input)
            loss = criterion(outputs, pt_cloud_target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        print(f"Epoch {epoch+1} finished. Time taken: {end_time - start_time:.2f}s, Loss: {loss.item():.4f}")

    print("Training complete!")
    save_model(model)

if __name__ == "__main__":
    train_model()