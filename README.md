3D Point Cloud Generation from a Single Image:

This project implements a deep learning model to reconstruct a 3D point cloud from a single 2D image. 
The approach is inspired by the paper "A Point Set Generation Network for 3D Object Reconstruction from a Single Image".

The network takes a 2D image as input, passes it through an encoder to extract features, and then uses
a decoder to generate the 3D coordinates of a point cloud.

Model Architecture
The model is composed of a 2D convolutional encoder that processes the input image and 
a fully connected decoder that generates the 3D point cloud.

<img width="568" height="243" alt="image" src="https://github.com/user-attachments/assets/c0709c3c-8a23-49cd-8a3c-7deb321299a9" />

utput Example
The model predicts the 3D coordinates for a set of points, which can then be rendered to visualize the reconstructed 3D object.
The following image shows an example of the model's output.

<img width="280" height="156" alt="image" src="https://github.com/user-attachments/assets/d568f199-7d76-4828-934c-043296f4e7a8" />

Reference "https://arxiv.org/pdf/1612.00603"

