import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_layers(x)

class Decoder(nn.Module):
    def __init__(self, num_points=1024):
        super(Decoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 48 * 64, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, num_points * 3) # Output is N*3 for the point cloud
        )
        self.num_points = num_points

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_layers(x)
        return x.view(batch_size, self.num_points, 3)

class PointCloudGenerationModel(nn.Module):
    def __init__(self):
        super(PointCloudGenerationModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded