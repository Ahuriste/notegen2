import tensorflow as tf
import numpy as np

def build_and_compile ():
	model = tf.keras.Sequential([
		tf.keras.layers.LSTM(64, input_shape=(10, 130)),
		tf.keras.layers.Dense(130)
	])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model