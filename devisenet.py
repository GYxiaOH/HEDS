"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np


class DeViSENet(object):
    """Implementation of the VggsNet."""

    def __init__(self, x,drop_rate,is_training):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.Dropout = drop_rate
        self.Is_training = is_training

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        
        conv6_1000 = tf.layers.conv2d(self.X,256,5,activation=tf.nn.relu,padding='same',name='conv6_1000',kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
        pool6_1000 = tf.layers.max_pooling2d(conv6_1000,3,3,name='pool6_1000',padding='same')
        conv6_200 = tf.layers.conv2d(self.X, 256, 5, activation=tf.nn.relu, padding='same', name='conv6_200',kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
        pool6_200 = tf.layers.max_pooling2d(conv6_200, 3, 3, name='pool6_200',padding='same')
        conv6_10 = tf.layers.conv2d(self.X, 256, 5, activation=tf.nn.relu, padding='same', name='conv6_10',kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
        pool6_10 = tf.layers.max_pooling2d(conv6_10, 3, 3, name='pool6_10',padding='same')
        fc7_1000 = tf.contrib.layers.flatten(pool6_1000)
        fc7_1000 = tf.layers.dense(fc7_1000,4096,activation=tf.nn.relu,name='fc7_1000',kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01),
                                   bias_initializer=tf.ones_initializer())
        fc7_1000 = tf.layers.dropout(fc7_1000,rate=self.Dropout)
        fc8_1000 = tf.layers.dense(fc7_1000,1000,activation=tf.nn.relu,name='fc8_1000',kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
        fc7_200 = tf.contrib.layers.flatten(pool6_200)
        fc7_200 = tf.layers.dense(fc7_200, 4096, activation=tf.nn.relu, name='fc7_200',kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01),
                                  bias_initializer=tf.ones_initializer())
        fc7_200 = tf.layers.dropout(fc7_200, rate=self.Dropout)
        fc8_200 = tf.layers.dense(fc7_200, 200, activation=tf.nn.relu, name='fc8_200',kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
        fc7_10 = tf.contrib.layers.flatten(pool6_10)
        fc7_10 = tf.layers.dense(fc7_10, 4096, activation=tf.nn.relu, name='fc7_10',kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01),
                                 bias_initializer=tf.ones_initializer())
        fc7_10 = tf.layers.dropout(fc7_10, rate=self.Dropout)
        fc8_10 = tf.layers.dense(fc7_10, 10, activation=tf.nn.relu, name='fc8_10',kernel_initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
        self.fc8_1000 = fc8_1000
        self.fc8_10 = fc8_10
        self.fc8_200 = fc8_200

    #if b or t smaller than  0,it will out NAN
    def sigmoidnew(self,x):
        with tf.variable_scope("sigmoidnew", reuse=tf.AUTO_REUSE):
            b = tf.get_variable(name="b", dtype=tf.float32,
                                initializer=tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32), trainable=True)
            t = tf.get_variable(name="t", dtype=tf.float32,
                                initializer=tf.random_uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32), trainable=True)
            b = tf.clip_by_value(b, 0, 1)
            t = tf.clip_by_value(t, 0, 1)
            alpha = 2. / b * tf.log((1 - t) / t)
            rx = 1. / (1 + tf.exp(-alpha * (x - b / 2.)))
        return rx