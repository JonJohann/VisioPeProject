import torch
import torch.nn as nn
from utils import projection

"""
The discriminator convolves an image, projects it several times through convolutions before finally taking a sigmoid, deciding if the image 
is real or fake
"""
class Discriminator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            projection(out_channels, out_channels * 2, 4, 2, 1, False),
            projection(out_channels * 2, out_channels * 4, 4, 2, 1, False),
            projection(out_channels * 4, out_channels * 8, 4, 2, 1, False),
            nn.Conv2d(out_channels * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.discriminator(x)


"""
The generator takes in a vector from the normal distribution and outputs a 2D vector
"""
class Generator(nn.Module):
    def __init__(self, in_channels: int, channels_img: int, features_g: int):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            projection(in_channels, features_g * 16, 4, 1, 0, True),
            projection(features_g * 16, features_g * 8, 4, 2, 1, True),
            projection(features_g * 8, features_g * 4, 4, 2, 1, True),
            projection(features_g * 4, features_g * 2, 4, 2, 1, True),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.generator(x)
