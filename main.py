from nn import NeuralNetwork
import numpy as np
import pandas as pd

colors = np.genfromtxt("./training.csv", delimiter=',')
textColors = np.genfromtxt("./supervised.csv", delimiter=',')

x = colors[1:]

x = np.divide(x, 255)
y = textColors[1:]

nn = NeuralNetwork(x,y)

#nn.train(1500)