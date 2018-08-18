from nn import NeuralNetwork
import numpy as np
from matplotlib import pyplot

colors = np.genfromtxt("./training.csv", delimiter=',')
textColors = np.genfromtxt("./supervised.csv", delimiter='')

x = colors[1:]
x = np.divide(x, 255)

y = np.array([textColors[1:]])
y = np.transpose(y)

nn = NeuralNetwork(x,y)

results = nn.train(3000)

keys = list(results.keys())
itens = [results[p] for p in keys]

print(pyplot.plot(keys, itens))