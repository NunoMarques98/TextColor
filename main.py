from nn import NeuralNetwork
import numpy as np
import pandas as pd

x = np.array([  [255,255,255],
                [0,0,0],
                [171, 35, 223],
                [122, 127, 76],
                [193, 116, 21],
                [21, 178, 193],
                [160, 123, 123],
                [196, 58, 58],
                [247, 144, 1]])

x = np.divide(x, 255)

y = np.array([[0],[1],[0],[1], [1], [0], [0], [1], [0]])

nn = NeuralNetwork(x,y)

nn.train(1500)