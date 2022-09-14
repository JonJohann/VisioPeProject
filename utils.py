"""
Utility functions for reusable modules and weight initializer
"""
import torch.nn as nn

def projection(in_channels: int, out_channels: int, kernel_size, stride, padding, frac_strided: bool):
    convolution = nn.ConvTranspose2d if frac_strided else nn.Conv2d
    return nn.Sequential(
        convolution(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU() if frac_strided else nn.LeakyReLU(0.2),
    )

def set_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)

