from typing import Tuple
import math

import torch
import torch.nn as nn

from models import Discriminator, Generator
from utils import generate_even_data


def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):
    input_length = int(math.log(max_int, 2)) #beskriver binærlengden på 128

    # Models: outputer to modeller med ant. nevroner lik som lengden på makstallet i binærform.
    generator = Generator(input_length) 
    discriminator = Discriminator(input_length)

    # Optimizers. finner vektene/parameterene med min.loss, med learning rate =0.001
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator
        # Need float type instead of int
        #genererer en tensor med tall mellom 0 og 2, der size beskriver
        #tensorens arkitetur. her da en liste med lister. der lengden av denne er batch size,
        #og hver listeverdi er en annen liste med tall mellom 0 og 2.
        # betyr da en lsite med lsite
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise) #en batch med noise inn i Generator, og output: sigmoid i etelerannet

        # her instansierer vi faktisk
        #true_labels: liste med 1ere
        #true_data: liste med selve even numbersene
        true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
        true_labels = torch.tensor(true_labels).float()
        true_data = torch.tensor(true_data).float()

        # Train the generator
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true.

        #1. discriminator tar inn generator random noise. vektene dårlige i starten, får en output
        #2, loss til discriminator sin out med random noise ift. true labels
        #3.og4. oppdater vektene tilk generator ift. discriminator sin loss
        #hvis discriminator sin loss er liten, nærme 0, betyr at gjettet label er nærme true label,
        #og dermed har den gjort feil, og det ønsker jo generatoren, derfor ønsker vi å oppdatere
        #dens optimizer med et lite tall

        #hvis discriminator sin loss er stor, nærme 1, betyr det at den har gjettet feil fra true label
        #og dermed ønsker vi å oppdatere generator så mye som mulig, og oppdaterer med dens tall
        generator_discriminator_out = discriminator(generated_data) #in: sigmoid gjort noe med noise. out: gir discriminator sin verdi til random noise.
        generator_loss = loss(generator_discriminator_out, true_labels) #her begynner backwardpropogation. finner ut loss i discriminator med noisen ift. true labels.
        generator_loss.backward()
        generator_optimizer.step() #her oppdateres selve modellen

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()




    return generator, discriminator


if __name__ == "__main__":
    train()