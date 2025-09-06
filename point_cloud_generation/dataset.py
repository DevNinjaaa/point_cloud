import os
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
from zipfile import ZipFile
import requests

def download_and_extract_data(url, destination_dir):
    """Downloads and extracts a zip file to the specified directory."""
    zip_path = os.path.join(destination_dir, os.path.basename(url))
    if os.path.exists(destination_dir):
        print(f"Data directory '{destination_dir}' already exists. Skipping download.")
        return

    print(f"Downloading data from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Extracting to '{destination_dir}'...")
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_dir)

    os.remove(zip_path)
    print("Download and extraction complete.")

def prepare_datasets():
    # You will need to find the correct links for the dataset and a corresponding image dataset.
    # The ModelNet40 dataset is available from Princeton University.
    # A common source is the 2D views of ModelNet, such as those used in PointNet.
    # You may need to manually download and structure the image data to match the ply files.
    # For now, I will use placeholders for the download links.
    
    # Download and extract the ModelNet40 point cloud data
    modelnet_url = "correct the links"
    modelnet_dir = "correct the links"
    download_and_extract_data(modelnet_url, modelnet_dir)
    
    # Placeholder for the image dataset download. You will need to find and download this yourself.
    # The structure expected by the code is ./classes/{model_name}/{split}/{version}/{sample}.0.png
    # The user needs to manually find and structure this part of the dataset.
    
class CustomDataset(Dataset):
  def __init__(self, split, img_ds_path="./data/classes", pt_cloud_ds_path="./data/ModelNet40_ply"):
    self.img_ds_path = img_ds_path
    self.pt_cloud_ds_path = pt_cloud_ds_path
    self.split = split
    self.data = []

    models = os.listdir(img_ds_path)
    for model_name in models:
        model_versions = os.listdir(os.path.join(img_ds_path, model_name, split))
        for version in model_versions:
            sample = version.split(".")[0]
            img_path = os.path.join(img_ds_path, model_name, split, version, f"{sample}.0.png")
            point_cloud_path = os.path.join(pt_cloud_ds_path, model_name, split, f"{sample}.ply")
            
            if os.path.exists(img_path) and os.path.exists(point_cloud_path):
                self.data.append((img_path, point_cloud_path))
    
    self.resize_transform = torchvision.transforms.Resize((192, 256))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_path, pt_cloud_path = self.data[idx]

    pcd = o3d.io.read_point_cloud(pt_cloud_path)
    pcd_np_arr = np.asarray(pcd.points)
    
    # Randomly sample 1024 points from the point cloud
    idx = np.random.randint(0, pcd_np_arr.shape[0], size=1024)
    pt_cloud_arr = torch.tensor(pcd_np_arr[idx]).to(torch.float32)

    img_arr = read_image(img_path)
    img_resize = self.resize_transform(img_arr)
    
    # Normalize the point cloud and image data
    return img_resize.to(torch.float32) / 255.0, pt_cloud_arr / 500.0