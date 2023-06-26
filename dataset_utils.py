import tensorflow as tf
import numpy as np

def write (filename, inputs, targets):
# Save data to TFRecords file


    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(len(inputs)):
            feature = {
                'input': tf.train.Feature(float_list=tf.train.FloatList(value=inputs[i].flatten())),
                'target': tf.train.Feature(float_list=tf.train.FloatList(value=targets[i].flatten()))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

def load (filename):

    # Load the TFRecords file as a dataset
    def parse_example(example_proto):
        feature_description = {
            'input': tf.io.FixedLenFeature((10, 130), tf.float32),
            'target': tf.io.FixedLenFeature((130,), tf.float32)
        }
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        return parsed_example['input'], parsed_example['target']

    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_example)
    return dataset
