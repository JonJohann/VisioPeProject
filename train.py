
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from utils import set_weights
from models import Generator, Discriminator

# Initializing hyperparameters
epochs = 5
lr = 1e-4
batch_size = 128
img_size = 64
img_channels = 1
z_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
discriminator_size = 64
generator_size = 64

# Creating transforms
transforms = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

#Downloading the FashionMNIST dataset and setting up the dataloader
dataset = datasets.FashionMNIST(root="dataset/", train=True, transform=transforms,
                       download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initializing our two models
discriminator = Discriminator(img_channels, discriminator_size).to(device)
generator = Generator(z_dim, img_channels, generator_size).to(device)
set_weights(generator)
set_weights(discriminator)

# Sending our model parameters to the Adam optimizer
disc_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
gen_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
loss_function = nn.BCELoss()
discriminator.train()
generator.train()

for epoch in range(epochs):
    for batch_index, (real_image, _ ) in enumerate(dataloader):
        real_image = real_image.to(device)
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_image = generator(noise)

        # Discriminator training
        disc_image = discriminator(real_image).reshape(-1)
        disc_image_loss = loss_function(disc_image, torch.ones_like(disc_image))
        disc_fake_img = discriminator(fake_image.detach()).reshape(-1)
        disc_fake_loss = loss_function(disc_fake_img, torch.zeros_like(disc_fake_img))
        disc_loss = (disc_image_loss + disc_fake_loss) / 2 
        discriminator.zero_grad()
        disc_loss.backward()
        disc_optim.step()

        # Generator training
        disc_fake_image_after = discriminator(fake_image).reshape(-1)
        gen_loss = loss_function(disc_fake_image_after, torch.ones_like(disc_fake_image_after))
        generator.zero_grad()
        gen_loss.backward()
        gen_optim.step()

        if batch_index % 140 == 0:
          print(f'Epoch: [{epoch}/{epochs}] | Batch: [{batch_index}/{len(dataloader)}]')

# Visualizing our results
def show_images(interpolated: bool):
  n = 30
  interpolated_noise_vectors = torch.randn(n, z_dim, 1, 1)
  images = generator(interpolated_noise_vectors.to(device))
  if interpolated:
    first_noise_vector = torch.tensor(interpolated_noise_vectors[0])
    second_noise_vector = torch.tensor(interpolated_noise_vectors[n-1])
    for index, vector in enumerate(interpolated_noise_vectors):
      interpolated_noise_vectors[index] = first_noise_vector*(((n-1)-index)/(n-1)) + second_noise_vector*(index/(n-1))
  images = generator(interpolated_noise_vectors.to(device))
  fig = plt.figure(figsize=(4., 4.))
  grid = ImageGrid(fig, 111, nrows_ncols=(5, 6), axes_pad=0.1)

  for ax, im in zip(grid, images):
      ax.imshow(im.cpu().detach().numpy()[0], cmap="gray")

  plt.show()

# Change to false to show a random selection of 30 images
show_images(True)