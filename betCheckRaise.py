
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


import numpy as np
import pandas

np.random.seed(10)

dataset = pandas.read_csv('poker-hand-training-true.csv', header = None)

values = dataset.values[:,0:10].astype(int)
labels_num = dataset.values[:,10:11].astype(int)

#one-hot encoding
labels = to_categorical(labels_num)

#model

model = Sequential()
model.add ( Dense ( 20, input_dim = 10, activation ='relu' ) )
model.add ( Dense ( 40, activation = 'relu' ) )
model.add ( Dense ( 10, activation = 'softmax' ) )

#compling ( setup of the model )
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#fit trains the model
model.fit( values, labels, epochs=15, batch_size = 10)

#evaluate returns the loss and mertric values ( i.e here the accuracy)
scores = model.evaluate(values, labels)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
