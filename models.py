from typing import Optional
import math
import torch.nn as nn

#tomt nettverk med input_length som input(nevroner), og input_length som output(nevroner)
class Generator(nn.Module):
    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length)) #det som kommer inn er binærformen på et tall, derfor er dette antall nevroner på input og output. trenger nevroner for det største tallet
        self.activation = nn.Sigmoid()

    #definerer hvordan data skal flowe gjennom nettverekt
    #i dette tilfelle: sigmoid tar inn noe og spytter ut noe annet
    #in er i dette tilfelle linear_layer med x som input
    #på en måte selve realisingen av Generator
    def forward(self, x):
        return self.activation(self.dense_layer(x))

#tomt nettverk med input_length som input(nevroner), og 1 output(nevroner)
class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_length), 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))