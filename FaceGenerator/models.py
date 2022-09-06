import pathlib
import glob
#import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from keras import layers
import time
from keras.layers import Conv2D,Dropout,Dense,Flatten,Conv2DTranspose,BatchNormalization,LeakyReLU,Reshape
from skimage.color import rgb2gray


#data_dir =pathlib.Path(data_dir)
# print(os.listdir("pipeline"))
BUFFER_SIZE = 60000
BATCH_SIZE = 256

data_dir = "FaceGenerator/pipeline/Humans"
path_faces = []
for path in os.listdir(data_dir):

    if '.jpg' in path:
        path_faces.append(os.path.join(data_dir, path))
new_path = path_faces[0:2]

#PIL.Image.open(path).crop(crop)).
crop = (30, 55, 150, 175)
faces = [np.array((PIL.Image.open(path)).resize((64,64))) for path in new_path]
#grey = rgb2gray(faces)
grey = tf.image.rgb_to_grayscale(faces)
train_faces = np.array(grey, dtype=object).astype(np.float32)
train_faces = (train_faces - 127.5) / 127.5

plt.imshow(grey[0])
plt.show()
print(train_faces.shape)

BUFFER_SIZE = 60000
BATCH_SIZE = 256
train_dataset = tf.data.Dataset.from_tensor_slices(train_faces).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

for elem in train_dataset:
    print(elem)


#generator:
generator = tf.keras.Sequential()
#lag 1:
generator.add(Dense(1024*4*4, input_shape=(100,)))
generator.add(Reshape((4, 4, 1024)))

generator.add(Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))

generator.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())

generator.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())

generator.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same",
                                 activation='tanh'))

generator.summary()


#discriminator
discriminator= tf.keras.Sequential()
discriminator.add(Conv2D(32, kernel_size=4, strides=2, padding="same",input_shape=[64,64, 3]))
discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())

discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())

discriminator.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
discriminator.add(LeakyReLU(0.2))

discriminator.add(Flatten())
discriminator.add(Dropout(0.5))
discriminator.add(Dense(1,activation='sigmoid'))

discriminator.summary()

"""""
for i in range(len(grey)):
    grey[i] = ((grey[i] - grey[i].min())/(255 - grey[i].min()))

faces = np.array(grey, dtype=object)

 # shape:(x, 180, 180, 3) betyr x antall bilder
# der hvert bilde har 180x180 pixler, og dette er 3 ganger, pga 3 kanaler.


#print(faces[0])
#img =PIL.Image.open(path_faces[0])
#img.show()
#train_data = tf.data.Dataset.from_tensor_slices(grey).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
noiseshape = 100 


#generator:
generator = tf.keras.Sequential()
#lag 1:
generator.add(Dense(1024*4*4, input_shape=(noiseshape,)))
generator.add(Reshape((4, 4, 1024)))

generator.add(Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))

generator.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())

generator.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())

generator.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same",
                                 activation='tanh'))

generator.summary()


#discriminator
discriminator= tf.keras.Sequential()
discriminator.add(Conv2D(32, kernel_size=4, strides=2, padding="same",input_shape=[64,64, 3]))
discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())
discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())
discriminator.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Flatten())
discriminator.add(Dropout(0.5))
discriminator.add(Dense(1,activation='sigmoid'))


#train together:

noise = tf.random.normal([1, 100])
random_image = generator(noise, training=False)
decision = discriminator(random_image)
print (decision)



4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
	# save the generator model
	g_model.save('generator.h5')


"""""







#for path in os.listdir(data_dir):
 #   if '.jpg' or '.png' or '.jpeg' in path:
 #       path_faces.append(os.path.join(data_dir, path))

#print(len(path_faces))

#images = [np.array((PIL.Image.open(path).crop(crop)).resize((64,64))) 
#for path in new_path]




