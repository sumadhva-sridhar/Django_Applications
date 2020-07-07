import numpy as np
import pandas as pd
import tensorflow as tf

train_data = np.array([1,2,2,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,10], dtype = 'float')
train_labels = np.array([19,43,39,60,58,61,84,77,78,82,100,101,102,99,120,124,115,119,143,141,137,160,161,158,183,178,185,200,207,199,197], dtype = 'float')

model = tf.keras.Sequential ([
	tf.keras.layers.Dense(1, input_shape = [1])
])

model.compile (optimizer = 'sgd', loss = 'mean_squared_error')

model.fit(train_data, train_labels, epochs = 200, verbose = 0)

model.save("my_model")

print("Model saved")
