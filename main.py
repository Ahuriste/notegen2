import tensorflow as tf
import numpy as np
import dataset_utils as du
import model as md


# Create random input and target vectors
num_samples = 10
input_shape = (10, 130)
target_shape = (130,)
inputs = np.random.rand(num_samples, *input_shape)
targets = np.random.rand(num_samples, *target_shape)
du.write("data.tfrecords", inputs, targets)
dataset = du.load("data.tfrecords")
dataset = dataset.shuffle(buffer_size=10).batch(2)


model = md.build_and_compile ()

model.fit (dataset)

input = np.random.rand (1,*input_shape)
print(model.predict(input))
