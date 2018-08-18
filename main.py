from nn import NeuralNetwork
import numpy as np
from matplotlib import pyplot

colors = np.genfromtxt("./training.csv", delimiter=',')
textColors = np.genfromtxt("./supervised.csv", delimiter='')

x = colors[1:]
x = np.divide(x, 255)

y = np.array([textColors[1:]])
y = np.transpose(y)

nn = NeuralNetwork(x, y, 0.02)

results = nn.train(3000)

keys = list(results.keys())
itens = [results[p] for p in keys]

pyplot.plot(keys, itens)

pyplot.xlabel("Epochs")
pyplot.ylabel("Loss")

#pyplot.show()

print(nn.value([[209, 117, 23]]))