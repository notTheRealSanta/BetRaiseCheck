
from keras.models import Sequential
from keras.layers import Dense

import numpy as np 
import pandas

np.random.seed(10)

dataset = pandas.read_csv('poker-hand-training-true.csv', header = None)

print dataset