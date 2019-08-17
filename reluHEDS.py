"""Script to finetune AlexNet using Tensorflow.

HEDS,a deep learning about hierachical structure embedding

Author: GYxiaOH
contact: zhhy1994226@163.com
"""

import os
import sys
import math

import numpy as np
import tensorflow as tf
import argparse
import ast

from tensorflow.contrib import learn
from vggsnet import VggsNet
from devisenet import DeViSENet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from sklearn import preprocessing
from tensorflow.contrib.tensorboard.plugins import projector

"""
Configuration Part.
"""
### chose GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# parameter
parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("isTrain", help="True means it will train a model and False means load a npy to test",
                    type=ast.literal_eval, default=True)
args = parser.parse_args()

# Path to the textfiles for the trainings and validation set
source_train_file = '../oneshot/70/trainlmdb.txt'
target_train_file = './targettrain.txt'
source_val_file = './sourceval.txt'
target_val_file = './vallmdb.txt'
test_file = './testlmdb.txt'

# Path for pretrain model mean file and click feature
pretrain_path = '../prevgg.npy'
meanfile_path = './dog129sandt_mean.npy'
npypath = './speed/pretrain.npy'
click1000_path = '../oneshot/70/wordweightnorm.txt'
click200_path = '../oneshot/70/10-20/20/wordweight.txt'
click10_path = '../oneshot/70/10-20/10/wordweight.txt'

# global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(0.01, global_step,100,0.1,staircase=True)

# params
global_step = tf.Variable(0, trainable=False)
l1weight_decay = tf.train.exponential_decay(0.05, global_step, 1300, 0.2, staircase=True)
learning_rate = tf.train.exponential_decay(0.001, global_step, 1800, 0.1, staircase=True)
thresholdweight = 0.6
# threshold1000 = thresho5dweight * math.sqrt(1. / 1000)
# threshold200 = thresholdweight * math.sqrt(1. / 200)
# threshold10 = thresholdweight * math.sqrt(1. / 10)
# learning_rate = 0.001 #0.01
num_epochs = 50
batch_size = 64
save_epoch = 5

# Network params
dropout_rate = 0.1
num_classes = 129
skip_layers = ['fc8']  # fc8 7 6
train_layers = ['feature_sel', 'sigmoidnew', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6_10', 'fc7_10',
                'fc8_10', 'conv6_200', 'fc7_200', 'fc8_200', 'conv6_1000', 'fc7_1000', 'fc8_1000', 'fc9']

# How often we want to write the tf.summary data to disk
display_step = 20
validation_epoch = 5

# Path for tf.summary.FileWriter and to store model+- checkpoints
filewriter_path = "./DeViSE/tensorboard"
checkpoint_path = "./DeViSE/checkpoints"

#Max Test Accuracy
maxTestAccuracy = 0.

# TF placeholder for graph input and output
x_source = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
x_target = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
click_feature_1000 = tf.placeholder(tf.float32, [batch_size, 1000])
click_feature_200 = tf.placeholder(tf.float32, [batch_size, 200])
click_feature_10 = tf.placeholder(tf.float32, [batch_size, 10])

x = tf.concat(values=[x_source, x_target], axis=0)
s_class_labels = tf.placeholder(tf.float32, [batch_size, num_classes])
t_class_labels = tf.placeholder(tf.float32, [batch_size, num_classes])
drop_rate = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

# Initialize model
model_vgg = VggsNet(x, drop_rate, num_classes, skip_layers, pretrain_path)
model = DeViSENet(model_vgg.pool5, drop_rate, is_training)

print('Finish initialize model!')
# Link variable to model output
vec_1000 = model.fc8_1000
vec_200 = model.fc8_200
vec_10 = model.fc8_10

saver0 = tf.train.Saver(max_to_keep=6)

##############################################################
# feature selection initialization

feature_sel_init = np.load('./initialize.npy').item()
feature_sel_init_1000 = feature_sel_init['1000']
feature_sel_init_200 = feature_sel_init['200']
feature_sel_init_10 = feature_sel_init['10']


##############################################################
def scale(nums,scale,bias = 0):
    maxv = tf.nn.top_k(nums,1)[0][0][-1]
    minv = tf.nn.top_k(nums,nums.get_shape()[1])[0][0][-1]
    return tf.cond(tf.equal(maxv,minv),lambda :(tf.clip_by_value(nums,clip_value_min=1e-10,clip_value_max=1)*scale) + bias,lambda :tf.subtract(nums,minv)/(tf.subtract(maxv,minv)+1e-10)*scale + bias)

##############################################################
def load_npy(path, sess, allVariable):
    """load parameter from npy to test"""
    premodel = np.load(path).item()
    for key in premodel.keys():
        if key in allVariable:
            with tf.variable_scope(key.split('/')[0], reuse=tf.AUTO_REUSE):
                try:
                    weight = tf.get_variable(name='/'.join(key.split('/')[1:]), trainable=True)
                    sess.run(tf.assign(weight, premodel[key][0]))
                    print weight
                except:
                    print (key + "is a new variable")
        else:
            print (key + "is not in model")


##############################################################
# initialize data and save path
# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

if not os.path.isdir(filewriter_path):
    os.mkdir(filewriter_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    source_tr_data = ImageDataGenerator(source_train_file,
                                        mode='training',
                                        batch_size=batch_size,
                                        num_classes=num_classes,
                                        shuffle=True,
                                        mean=meanfile_path, need_c=True,
                                        click_fature_1000=click1000_path,
                                        click_fature_200=click200_path,
                                        click_fature_10=click10_path)
    target_tr_data = ImageDataGenerator(target_train_file,
                                        mode='training',
                                        batch_size=batch_size,
                                        num_classes=num_classes,
                                        shuffle=True,
                                        mean=meanfile_path)
    source_val_data = ImageDataGenerator(source_val_file,
                                         mode='inference',
                                         batch_size=batch_size,
                                         num_classes=num_classes,
                                         shuffle=False,
                                         mean=meanfile_path)
    target_val_data = ImageDataGenerator(target_val_file,
                                         mode='inference',
                                         batch_size=batch_size,
                                         num_classes=num_classes,
                                         shuffle=False,
                                         mean=meanfile_path)
    tst_data = ImageDataGenerator(test_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False,
                                  mean=meanfile_path)

    # create an reinitializable iterator given the dataset structure
    source_iterator = Iterator.from_structure(source_tr_data.data.output_types,
                                              source_tr_data.data.output_shapes)
    target_iterator = Iterator.from_structure(target_tr_data.data.output_types,
                                              target_tr_data.data.output_shapes)
    source_next_batch = source_iterator.get_next()
    target_next_batch = target_iterator.get_next()

# Ops for initializing the two different iterators
source_training_init_op = source_iterator.make_initializer(source_tr_data.data)
target_training_init_op = target_iterator.make_initializer(target_tr_data.data)

source_validation_init_op = target_iterator.make_initializer(source_val_data.data)
target_validation_init_op = target_iterator.make_initializer(target_val_data.data)

testing_init_op = target_iterator.make_initializer(tst_data.data)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(target_tr_data.data_size / batch_size))
source_val_batches_per_epoch = int(np.floor(source_val_data.data_size / batch_size))
target_val_batches_per_epoch = int(np.floor(target_val_data.data_size / batch_size))
test_batches_per_epoch = int(np.floor(tst_data.data_size / batch_size))

print('Start!')

##############################################################


# def f1():
#     return tf.get_variable(initializer=tf.zeros([1]))
# def f2():
#     return s_weight
# def constrain(X):
#     tf.cond(tf.less(s_weight,tf.constant(0)),f1(),f2())
# Loss and Accuracy
vec_10_source = tf.split(vec_10, num_or_size_splits=2, axis=0)[0]
vec_10_target = tf.split(vec_10, num_or_size_splits=2, axis=0)[1]
vec_200_source = tf.split(vec_200, num_or_size_splits=2, axis=0)[0]
vec_200_target = tf.split(vec_200, num_or_size_splits=2, axis=0)[1]
vec_1000_source = tf.split(vec_1000, num_or_size_splits=2, axis=0)[0]
vec_1000_target = tf.split(vec_1000, num_or_size_splits=2, axis=0)[1]

feature_sel_list = []
with tf.variable_scope("feature_sel", reuse=tf.AUTO_REUSE):
    #inin wiht 1
    # sel_1000 = tf.get_variable(name="sel_1000",shape=[1,1000],initializer=tf.ones_initializer)
    # sel_200 = tf.get_variable(name="sel_200", shape=[1,200], initializer=tf.ones_initializer)
    # sel_10 = tf.get_variable(name="sel_10", shape=[1,10], initializer=tf.ones_initializer)

    sel_1000 = tf.get_variable(name="sel_1000", initializer=feature_sel_init_1000,dtype=tf.float32)
    sel_200 = tf.get_variable(name="sel_200", initializer=feature_sel_init_200,dtype=tf.float32)
    sel_10 = tf.get_variable(name="sel_10", initializer=feature_sel_init_10,dtype=tf.float32)

    tf.add_to_collection("sel_weight", sel_10)
    tf.add_to_collection("sel_weight", sel_200)
    tf.add_to_collection("sel_weight", sel_1000)

    # sel_10 = tf.nn.leaky_relu(tf.subtract(sel_10,threshold1000),alpha=0.0001*threshold1000)
    # sel_200 = tf.nn.leaky_relu(tf.subtract(sel_200, threshold200),alpha=0.0001*threshold200)
    # sel_1000 = tf.nn.leaky_relu(tf.subtract(sel_1000, threshold10),alpha=0.0001*threshold10)

    # after_sel_1000 = tf.nn.relu(tf.subtract(tf.abs(tf.nn.l2_normalize(sel_1000, dim=1)), threshold1000))
    # after_sel_200 = tf.nn.relu(tf.subtract(tf.abs(tf.nn.l2_normalize(sel_200, dim=1)), threshold200))
    # after_sel_10 = tf.nn.relu(tf.subtract(tf.abs(tf.nn.l2_normalize(sel_10, dim=1)), threshold10))

    feature_sel_list.append(sel_1000)
    feature_sel_list.append(sel_10)
    feature_sel_list.append(sel_200)

vec_10_source = tf.layers.batch_normalization(vec_10_source, training=is_training, axis=1, renorm_momentum=0.9)
vec_10_target = tf.layers.batch_normalization(vec_10_target, training=is_training, axis=1, renorm_momentum=0.9)

vec_200_source = tf.layers.batch_normalization(vec_200_source, training=is_training, axis=1, renorm_momentum=0.9)
vec_200_target = tf.layers.batch_normalization(vec_200_target, training=is_training, axis=1, renorm_momentum=0.9)

vec_1000_source = tf.layers.batch_normalization(vec_1000_source, training=is_training, axis=1, renorm_momentum=0.9)
vec_1000_target = tf.layers.batch_normalization(vec_1000_target, training=is_training, axis=1, renorm_momentum=0.9)

# sel with l1
# vec_1000_source = tf.multiply(vec_1000_source, tf.tile(sel_1000, [batch_size, 1]))
# vec_1000_target = tf.multiply(vec_1000_target, tf.tile(sel_1000, [batch_size, 1]))
# vec_200_source = tf.multiply(vec_200_source, tf.tile(sel_200, [batch_size, 1]))
# vec_200_target = tf.multiply(vec_200_target, tf.tile(sel_200, [batch_size, 1]))
# vec_10_source = tf.multiply(vec_10_source, tf.tile(sel_10, [batch_size, 1]))
# vec_10_target = tf.multiply(vec_10_target, tf.tile(sel_10, [batch_size, 1]))

# sel with relu and l1
sel_1000 = scale(tf.abs(sel_1000),2,0)
sel_200 = scale(tf.abs(sel_200),2,0)
sel_10 = scale(tf.abs(sel_10),2,0)

sel_1000_minv = tf.nn.top_k(sel_1000, sel_1000.get_shape()[1])[0][0][-1]
sel_200_minv = tf.nn.top_k(sel_200, sel_200.get_shape()[1])[0][0][-1]
sel_10_minv = tf.nn.top_k(sel_10, sel_10.get_shape()[1])[0][0][-1]

sel_1000_meanv = tf.reduce_mean(sel_1000, 1)
sel_200_meanv = tf.reduce_mean(sel_200, 1)
sel_10_meanv = tf.reduce_mean(sel_10, 1)

dele1000 = thresholdweight*(sel_1000_meanv - sel_1000_minv)+sel_1000_minv
dele200 = thresholdweight*(sel_200_meanv - sel_200_minv)+sel_200_minv
dele10 = thresholdweight*(sel_10_meanv - sel_10_minv)+sel_10_minv


vec_1000_source = tf.multiply(vec_1000_source,tf.tile(tf.nn.relu(tf.subtract(sel_1000, dele1000)),[batch_size,1]))
vec_1000_target = tf.multiply(vec_1000_target, tf.tile(tf.nn.relu(tf.subtract(sel_1000, dele1000)),[batch_size,1]))
vec_200_source = tf.multiply(vec_200_source, tf.tile(tf.nn.relu(tf.subtract(sel_200, dele200)),[batch_size,1]))
vec_200_target = tf.multiply(vec_200_target, tf.tile(tf.nn.relu(tf.subtract(sel_200, dele200)),[batch_size,1]))
vec_10_source = tf.multiply(vec_10_source, tf.tile(tf.nn.relu(tf.subtract(sel_10, dele10)),[batch_size,1]))
vec_10_target = tf.multiply(vec_10_target, tf.tile(tf.nn.relu(tf.subtract(sel_10,dele10)),[batch_size,1]))


vec_class_source = tf.concat([vec_1000_source, vec_200_source, vec_10_source], axis=1)
vec_class_target = tf.concat([vec_1000_target, vec_200_target, vec_10_target], axis=1)

# vec_class = tf.nn.relu(vec_class,name="vec_class")click_feature

# fc9 = tf.layers.dense(vec_class,num_classes,name='fc9',kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.01))
#
# fc9_source = tf.split(fc9,num_or_size_splits=2,axis=0)[0]
# fc9_target = tf.split(fc9,num_or_size_splits=2,axis=0)[1]

# vec_class_source = tf.split(vec_class,num_or_size_splits=2,axis=0)[0]
# vec_class_target = tf.split(vec_class,num_or_size_splits=2,axis=0)[1]

fc9_source = tf.layers.dense(vec_class_source, num_classes, name='fc9',
                             kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                             reuse=tf.AUTO_REUSE)
fc9_target = tf.layers.dense(vec_class_target, num_classes, name='fc9',
                             kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                             reuse=tf.AUTO_REUSE)

source_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc9_source, labels=s_class_labels))
target_class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc9_target, labels=t_class_labels))
source_prediction = tf.nn.softmax(fc9_source)
target_prediction = tf.nn.softmax(fc9_target)

class_loss = 0.6 * source_class_loss + 0.9 * target_class_loss

binary_vec_1000 = model.sigmoidnew(vec_1000_source)
binary_vec_200 = model.sigmoidnew(vec_200_source)
binary_vec_10 = model.sigmoidnew(vec_10_source)

binary_1000 = model.sigmoidnew(click_feature_1000)
binary_200 = model.sigmoidnew(click_feature_200)
binary_10 = model.sigmoidnew(click_feature_10)

aaaaa = tf.trainable_variables()
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
# print tf.GraphKeys.WEIGHTS
print tf.trainable_variables()
weight_list = [v for v in var_list if v.name.split('/')[1] != 'biases:0' and v.name.split('/')[1] != 'bias:0' and v not in tf.get_collection("sel_weight")]
bias_list = [v for v in var_list if v.name.split('/')[1] == 'biases:0' or v.name.split('/')[1] == 'bias:0']
trainw_list = [v for v in var_list if v.name.split('/')[1] != 'biases:0' and v.name.split('/')[1] != 'bias:0']

l1regular = tf.contrib.layers.l1_regularizer(l1weight_decay)
l2regular = tf.contrib.layers.l2_regularizer(0.0005)
# aaaaalist = tf.get_collection("sel_weight")
l2regular_fea = tf.contrib.layers.l2_regularizer(0.0005)
regular_l2_fea = tf.contrib.layers.apply_regularization(l2regular_fea, weights_list=tf.get_collection("sel_weight"))

# regular_l1 = tf.contrib.layers.apply_regularization(l1regular, weights_list=tf.get_collection("sel_weight"))
regular = tf.contrib.layers.apply_regularization(l2regular, weights_list=weight_list)
regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

with tf.name_scope("prediction_loss"):
    greater_vec_1000_source = vec_1000_source!=0
    onezero_vec_1000_source = tf.where(greater_vec_1000_source,
                                       tf.ones_like(vec_1000_source),tf.zeros_like(vec_1000_source))

    greater_vec_200_source = vec_200_source != 0
    onezero_vec_200_source = tf.where(greater_vec_200_source, tf.ones_like(vec_200_source),
                                      tf.zeros_like(vec_200_source))

    greater_vec_10_source = vec_10_source != 0
    onezero_vec_10_source = tf.where(greater_vec_10_source, tf.ones_like(vec_10_source), tf.zeros_like(vec_10_source))

    onezero_vec_1000_source_b = tf.stop_gradient(onezero_vec_1000_source)
    onezero_vec_200_source_b = tf.stop_gradient(onezero_vec_200_source)
    onezero_vec_10_source_b = tf.stop_gradient(onezero_vec_10_source)

    click_feature_1000_useful = tf.multiply(onezero_vec_1000_source,click_feature_1000,name='click_feature_1000_useful')
    binary_vec_1000_t = tf.multiply(onezero_vec_1000_source,binary_1000,name='binary_vec_1000_t')
    binary_vec_1000_p = tf.multiply(onezero_vec_1000_source,model.sigmoidnew(vec_1000_source),name='binary_vec_1000_p')

    click_feature_200_useful = tf.multiply(onezero_vec_200_source,click_feature_200,name='click_feature_200_useful')
    binary_vec_200_t = tf.multiply(onezero_vec_200_source,binary_200,name='binary_vec_200_t')
    binary_vec_200_p = tf.multiply(onezero_vec_200_source,model.sigmoidnew(vec_200_source),name='binary_vec_200_p')

    click_feature_10_useful = tf.multiply(onezero_vec_10_source,click_feature_10,name='click_feature_10_useful')
    binary_vec_10_t = tf.multiply(onezero_vec_10_source,binary_10,name='binary_vec_10_t')
    binary_vec_10_p = tf.multiply(onezero_vec_10_source,model.sigmoidnew(vec_10_source),name='binary_vec_10_p')


    l2_loss_1000 = 1. / 2. * tf.reduce_mean(tf.reduce_sum(
        tf.square(tf.nn.l2_normalize(click_feature_1000_useful, dim=1) - tf.nn.l2_normalize(vec_1000_source, dim=1)), 1))
    binary_l2_loss_1000 = 1. / 2. * tf.reduce_mean(tf.reduce_sum(
        tf.square(
            tf.nn.l2_normalize(binary_vec_1000_t, dim=1) - tf.nn.l2_normalize(binary_vec_1000_p, dim=1)), 1))

    l2_loss_200 = 1. / 2. * tf.reduce_mean(tf.reduce_sum(
        tf.square(tf.nn.l2_normalize(click_feature_200_useful, dim=1) - tf.nn.l2_normalize(vec_200_source, dim=1)), 1))
    binary_l2_loss_200 = 1. / 2. * tf.reduce_mean(tf.reduce_sum(
        tf.square(tf.nn.l2_normalize(binary_vec_200_t, dim=1) - tf.nn.l2_normalize(binary_vec_200_p, dim=1)),
        1))

    l2_loss_10 = 1. / 2. * tf.reduce_mean(tf.reduce_sum(
        tf.square(tf.nn.l2_normalize(click_feature_10_useful, dim=1) - tf.nn.l2_normalize(vec_10_source, dim=1)), 1))
    binary_l2_loss_10 = 1. / 2. * tf.reduce_mean(tf.reduce_sum(
        tf.square(tf.nn.l2_normalize(binary_vec_10_t, dim=1) - tf.nn.l2_normalize(binary_vec_10_p, dim=1)),
        1))

    l2_loss = 0.8 * l2_loss_1000 + 0.04 * l2_loss_200 + 0.16 * l2_loss_10
    binary_l2_loss = 0.8 * binary_l2_loss_1000 + 0.04 * binary_l2_loss_200 + 0.16 * binary_l2_loss_10
    pre_loss = 0.9 * l2_loss + 0.1 * binary_l2_loss
loss_op = class_loss

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op1 = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(1 * loss_op + pre_loss + regularization_loss,
                                                                                               var_list=trainw_list,
                                                                                               global_step=global_step)
    train_op2 = tf.train.MomentumOptimizer(learning_rate=learning_rate * 2, momentum=0.9).minimize(
        1 * loss_op + pre_loss, var_list=bias_list)
    train_op = tf.group(train_op1, train_op2)

    # train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss_op)
with tf.control_dependencies(update_ops):
    target_correct_num = tf.equal(tf.argmax(target_prediction, 1), tf.argmax(t_class_labels, 1))
    target_accuracy = tf.reduce_mean(tf.cast(target_correct_num, tf.float32))

    source_correct_num = tf.equal(tf.argmax(source_prediction, 1), tf.argmax(s_class_labels, 1))
    source_accuracy = tf.reduce_mean(tf.cast(source_correct_num, tf.float32))

init = tf.global_variables_initializer()

##############################################################

#############################################################
writer = tf.summary.FileWriter(filewriter_path)

# # Add gradients to summary
# for gradient, var in gradients:
#     tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.add_to_collection("train_summary", tf.summary.histogram(var.name, var))

# Add the loss to summary

tf.add_to_collection("train_summary", tf.summary.scalar('class_loss', loss_op))
tf.add_to_collection("train_summary", tf.summary.scalar('l2_loss', l2_loss))
tf.add_to_collection("train_summary", tf.summary.scalar('binary_l2_loss', pre_loss))

# Add the accuracy to summary


# Merge  train_summaries together
train_summary = tf.summary.merge(tf.get_collection("train_summary"))
# merged_summary = tf.summary.merge_all()


source_acc = tf.get_variable(name="source_acc", initializer=tf.constant(0.0), dtype=tf.float32)
target_acc = tf.get_variable(name="target_acc", initializer=tf.constant(0.0), dtype=tf.float32)

source_acc_sum = tf.summary.scalar(name="source_acc_sum", tensor=source_acc)
target_acc_sum = tf.summary.scalar(name="target_acc_sum", tensor=target_acc)

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

saver = tf.train.Saver()

# validation accuracy summary

##############################################################
"""
Main Part of the finetuning Script.
"""
sel_show = {}
with tf.Session() as sess:
    if args.isTrain:
        sess.run(init)

        writer.add_graph(sess.graph)
        model_vgg.load_initial_weights(sess)

        for epoch in range(num_epochs):
            sess.run(source_training_init_op)
            sess.run(target_training_init_op)
            for step in range(train_batches_per_epoch):
                source_img_batch, source_label_batch, click_feature_1000_batch, click_feature_200_batch, click_feature_10_batch = sess.run(
                    source_next_batch)
                target_img_batch, target_label_batch = sess.run(target_next_batch)

                _, loss, pre_los, l2_l, binary_l2_l, summary, decay, rate= sess.run(
                    [train_op, loss_op, pre_loss, l2_loss, binary_l2_loss, train_summary, l1weight_decay, learning_rate], feed_dict={x_source: source_img_batch,
                                              x_target: target_img_batch,
                                              click_feature_1000: click_feature_1000_batch,
                                              click_feature_200: click_feature_200_batch,
                                              click_feature_10: click_feature_10_batch,
                                              s_class_labels: source_label_batch,
                                              t_class_labels: target_label_batch,
                                              drop_rate: dropout_rate,
                                              is_training: True})

                if (step + 1) % display_step == 0 or (step + 1) == train_batches_per_epoch:
                    print("Epoch " + str(epoch + 1) + " , Step " + str(step + 1) + " , Learning Rate= " + str(
                        rate) + " , L1weight Decay= " + str(decay) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Prediction Loss= " + "{:.4f}".format(pre_los) + \
                          ", L2 Loss= " + "{:.4f}".format(l2_l) + ", Binary L2 Loss= " + "{:.4f}".format(binary_l2_l))
                    writer.add_summary(summary, epoch * train_batches_per_epoch + step)

            if (epoch + 1) % validation_epoch == 0:
                sess.run(source_validation_init_op)
                s_val_acc = 0
                t_val_acc = 0
                for step in range(source_val_batches_per_epoch):
                    source_img_batch, source_label_batch = sess.run(target_next_batch)
                    target_img_batch, target_label_batch = source_img_batch, source_label_batch
                    loss, sacc, tacc = sess.run([loss_op, source_accuracy, target_accuracy],
                                                feed_dict={x_source: source_img_batch,
                                                           x_target: target_img_batch,
                                                           click_feature_1000: np.zeros(shape=[64, 1000], dtype=float),
                                                           click_feature_200: np.zeros(shape=[64, 200], dtype=float),
                                                           click_feature_10: np.zeros(shape=[64, 10], dtype=float),
                                                           s_class_labels: source_label_batch,
                                                           t_class_labels: target_label_batch,
                                                           drop_rate: dropout_rate,
                                                           is_training: False})
                    s_val_acc += sacc

                print("Epoch " + str(epoch + 1) + ", Loss " + "{:.4f}".format(loss) + ", Souce Accuracy= " + \
                      "{:.4f}".format(s_val_acc / source_val_batches_per_epoch))
                sess.run(tf.assign(source_acc, (s_val_acc / source_val_batches_per_epoch)))
                summary = sess.run(source_acc_sum)
                writer.add_summary(summary, epoch)

            if (epoch + 1) % validation_epoch == 0:
                sess.run(target_validation_init_op)
                t_val_acc = 0
                for step in range(target_val_batches_per_epoch):
                    target_img_batch, target_label_batch = sess.run(target_next_batch)
                    source_img_batch, source_label_batch = target_img_batch, target_label_batch
                    loss, tacc = sess.run([loss_op, target_accuracy], feed_dict={x_source: source_img_batch,
                                                                                 x_target: target_img_batch,
                                                                                 click_feature_1000: np.zeros(
                                                                                     shape=[64, 1000], dtype=float),
                                                                                 click_feature_200: np.zeros(
                                                                                     shape=[64, 200], dtype=float),
                                                                                 click_feature_10: np.zeros(
                                                                                     shape=[64, 10], dtype=float),
                                                                                 s_class_labels: source_label_batch,
                                                                                 t_class_labels: target_label_batch,
                                                                                 drop_rate: dropout_rate,
                                                                                 is_training: False})

                    t_val_acc += tacc

                print("Epoch " + str(epoch + 1) + ", Loss " + "{:.4f}".format(loss) + ", target Accuracy= " + \
                      "{:.4f}".format(t_val_acc / target_val_batches_per_epoch))
                sess.run(tf.assign(target_acc, (t_val_acc / target_val_batches_per_epoch)))
                summary = sess.run(target_acc_sum)
                writer.add_summary(summary, epoch)
            if (epoch + 1) % save_epoch == 0:
                print("{} Start Test Before Save".format(datetime.now()))
                sess.run(testing_init_op)
                t_acc = 0
                for step in range(test_batches_per_epoch):
                    target_img_batch, target_label_batch = sess.run(target_next_batch)
                    source_img_batch, source_label_batch = target_img_batch, target_label_batch
                    loss, tacc, source_sel, target_sel = sess.run(
                        [loss_op, target_accuracy, vec_class_source, vec_class_target],
                        feed_dict={x_source: source_img_batch,
                                   x_target: target_img_batch,
                                   click_feature_1000: np.zeros(shape=[64, 1000],
                                                                dtype=float),
                                   click_feature_200: np.zeros(shape=[64, 200],
                                                               dtype=float),
                                   click_feature_10: np.zeros(shape=[64, 10],
                                                              dtype=float),
                                   s_class_labels: source_label_batch,
                                   t_class_labels: target_label_batch,
                                   drop_rate: dropout_rate,
                                   is_training: False})

                    t_acc += tacc
                print("Epoch " + str(epoch + 1) + ", Loss " + "{:.4f}".format(loss) + ", Target Test Accuracy= " + \
                      "{:.4f}".format(t_acc / test_batches_per_epoch))
                if ((t_acc / test_batches_per_epoch)>maxTestAccuracy):
                    maxTestAccuracy = (t_acc / test_batches_per_epoch)
                    sel_show['vec_class_source'] = source_sel
                    sel_show['vec_class_target'] = target_sel


                print("{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(checkpoint_path,
                                               'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)

                print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                               checkpoint_name))

        print("{} Start Test".format(datetime.now()))
        sess.run(testing_init_op)
        t_acc = 0
        for step in range(test_batches_per_epoch):
            target_img_batch, target_label_batch = sess.run(target_next_batch)
            source_img_batch, source_label_batch = target_img_batch, target_label_batch
            loss, tacc, source_sel, target_sel = sess.run([loss_op, target_accuracy, vec_class_source, vec_class_target],
                                                          feed_dict={x_source: source_img_batch,
                                                                     x_target: target_img_batch,
                                                                     click_feature_1000: np.zeros(shape=[64, 1000],
                                                                                                  dtype=float),
                                                                     click_feature_200: np.zeros(shape=[64, 200],
                                                                                                 dtype=float),
                                                                     click_feature_10: np.zeros(shape=[64, 10],
                                                                                                dtype=float),
                                                                     s_class_labels: source_label_batch,
                                                                     t_class_labels: target_label_batch,
                                                                     drop_rate: dropout_rate,
                                                                     is_training: False})

            t_acc += tacc
        print("Epoch " + str(epoch + 1) + ", Loss " + "{:.4f}".format(loss) + ", target Accuracy= " + \
              "{:.4f}".format(t_acc / test_batches_per_epoch))
        if ((t_acc / test_batches_per_epoch) > maxTestAccuracy):
            maxTestAccuracy = (t_acc / test_batches_per_epoch)
            sel_show['vec_class_source'] = source_sel
            sel_show['vec_class_target'] = target_sel
        np.save('./selshow.npy', sel_show)
        print("MaxTargetTest= " + "{:.4f}".format(maxTestAccuracy))

    else:
        sess.run(init)
        print [tf.global_variables()]
        load_npy(npypath, sess, [v._shared_name for v in tf.global_variables()])

        print("{} Start Test".format(datetime.now()))
        sess.run(testing_init_op)
        t_acc = 0
        for step in range(test_batches_per_epoch):
            target_img_batch, target_label_batch = sess.run(target_next_batch)
            source_img_batch, source_label_batch = target_img_batch, target_label_batch
            loss, tacc,vec10 = sess.run([loss_op, target_accuracy,vec_10_source], feed_dict={x_source: source_img_batch,
                                                                         x_target: target_img_batch,
                                                                         click_feature_1000: np.zeros(shape=[64, 1000],
                                                                                                      dtype=float),
                                                                         click_feature_200: np.zeros(shape=[64, 200],
                                                                                                     dtype=float),
                                                                         click_feature_10: np.zeros(shape=[64, 10],
                                                                                                    dtype=float),
                                                                         s_class_labels: source_label_batch,
                                                                         t_class_labels: target_label_batch,
                                                                         drop_rate: dropout_rate,
                                                                         is_training: False})

            t_acc += tacc
	    print vec10

        print(", Loss " + "{:.4f}".format(loss) + ", target Accuracy= " + \
              "{:.4f}".format(t_acc / test_batches_per_epoch))
#
#
# #learning_rate = tf.placeholder(tf.float32)
#
# t_label = tf.placeholder(tf.float32, [batch_size, w2v_dim])
# t_j = tf.placeholder(tf.float32, [batch_size, num_classes, w2v_dim])
# w2v_embd =  tf.placeholder(tf.float32,[num_classes,w2v_dim])
#
# devise_model = DeViSENet(score, w2v_dim)
# M = devise_model.M
#
# # List of trainable variables of the layers we want to train
# var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
#
# # Hinge loss function
# def get_max_margin(index):
#     margin = 0.1
#     max_margin = 0
#     score1 = tf.matmul(tf.reshape(t_label[index],[1,w2v_dim]),tf.reshape(M[index],[w2v_dim,1]))
#     for i in range(t_j.shape[1]):
#         score2 = tf.matmul(tf.reshape(t_j[index, i, :],[1,w2v_dim]),tf.reshape(M[index],[w2v_dim,1]))
#         max_margin += tf.maximum(0.0, margin - score1 + score2)
#     return max_margin
#
# with tf.name_scope("max_margin"):
#     margin_loss = 0
#     score1s = 0
#     for i in range(batch_size):
#         loss = get_max_margin(i)
#         margin_loss += loss
#     margin_loss = tf.reduce_mean(margin_loss / batch_size)
#
# # Op for calculating the loss(No use)
# with tf.name_scope("cross_ent"):
#     crossloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
#                                                                   labels=y))
#
# # Train op
# with tf.name_scope("train"):
#     # Get gradients of all trainable variables
# #   allloss = margin_loss +crossloss
#     gradients = tf.gradients(margin_loss, var_list)
#     gradients = list(zip(gradients, var_list))
#
#     # Create optimizer and apply gradient descent to the trainable variables
#     optimizer = tf.train.AdagradOptimizer(learning_rate)
#     train_op = optimizer.apply_gradients(grads_and_vars=gradients)
# #    train_op = optimizer.minimize(margin_loss)
# with tf.name_scope("accuracy"):
#     y_ = tf.argmax(y,1)
#     similarity = tf.matmul(M,tf.transpose(w2v_embd))
#     correct_num = tf.nn.in_top_k(similarity,y_,1)
#     accuracy = tf.reduce_mean(tf.cast(correct_num,tf.float32),name="accend")
#
# # Add gradients to summary
# for gradient, var in gradients:
#     tf.summary.histogram(var.name + '/gradient', gradient)
#
# # Add the variables we train to the summary
# for var in var_list:
#     tf.summary.histogram(var.name, var)
#
# # Add the loss to summary
# tf.summary.scalar('max_margin_loss', margin_loss)
# # Add the accuracy to summary
# tf.summary.scalar('accuracy',accuracy)
#
# # Merge all summaries together
# merged_summary = tf.summary.merge_all()
#
# # Initialize the FileWriter
# writer = tf.summary.FileWriter(filewriter_path)
#
# saver = tf.train.Saver()
#
# #validation accuracy summary
# val_acc_var = tf.get_variable('val_acc',
#                      dtype=tf.float32,
#                      initializer=tf.constant(0.0))
# val_acc =tf.summary.scalar('val_acc_var',x)
#
# ############################################################3
#
# # Start Tensorflow session
# with tf.Session() as sess:
#     total_loss = 0.
#     train_count = 0
#     epoch = 0
#     # Initialize all variables
#     sess.run(tf.global_variables_initializer())
#
#     # Add the model graph to TensorBoard
#     writer.add_graph(sess.graph)
#
#     # To continue training from one of your checkpoints
#
#     ckpt  = tf.train.get_checkpoint_state(checkpoint_path)
#
#
#     if not isTrain:
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, ckpt.model_checkpoint_path)
#             print (ckpt.model_checkpoint_path)
#         else:
#             print ('no test model')
#             sys.exit()
#
#
#
#     # Loop over number of epochs
#     if(isTrain):
#         #load pretrain model
#         ckpt1 = tf.train.get_checkpoint_state(checkpoint_path0)
#         if ckpt and ckpt.model_checkpoint_path:
#             saver.restore(sess, ckpt.model_checkpoint_path)
#             print (ckpt.model_checkpoint_path)
#         elif ckpt1 and ckpt1.model_checkpoint_path:
#             saver0.restore(sess, ckpt1.model_checkpoint_path)
#             print (ckpt1.model_checkpoint_path)
#         else:
#             model.load_initial_weights(sess)
#             print ('initialize with npy')
#         print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
#                                                           filewriter_path))
#         print("{} Start training...".format(datetime.now()))
#         for epoch in range(num_epochs):
#
#             print("{} Epoch number: {}".format(datetime.now(), epoch+1))
#             sess.run(training_init_op)
#             loss = 0.
#             #y_score = 0.
#             #total_score = 0
#
#             for step in range(train_batches_per_epoch):
#
#                 img_batch, label_batch = sess.run(next_batch)
#
#
#
#                 y_label = sess.run(tf.argmax(label_batch, 1))
#
#                 t_label_batch = np.zeros(shape=(batch_size, w2v_dim))
#                 t_j_batch = np.zeros(shape=(batch_size, num_classes, w2v_dim))
#                 for i in range(batch_size):
#                     for j in range(num_classes):
#                         if j == y_label[i]:
#                             t_label_batch[i]= get_normalize_w2v(j)
#                         else:
#                             t_j_batch[i][j] = get_normalize_w2v(j)
#
#
#                 _,loss= sess.run([train_op,margin_loss],
#                                                  feed_dict={x: img_batch,
#                                                  y: label_batch,
#                                                  keep_prob: 1.,
#                                                  t_label: t_label_batch,
#                                                  t_j: t_j_batch,
#                                                  w2v_embd:label_w2v})
#
#                 total_loss += loss
#                 train_count += 1
#                 if step % display_step == 0:
#                     acc = sess.run(accuracy,feed_dict={x: img_batch,
#                                                  y: label_batch,
#                                                  keep_prob: 1.,
#                                                  t_label: t_label_batch,
#                                                  t_j: t_j_batch,
#                                                  w2v_embd:label_w2v})
#                     print("Step " + str(step) + ", Minibatch Loss= " + \
#                           "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                           "{:.3f}".format(acc))
#                     s = sess.run(merged_summary, feed_dict={x: img_batch,
#                                                             y: label_batch,
#                                                             keep_prob: 1.,
#                                                             t_label: t_label_batch,
#                                                             t_j: t_j_batch,
#                                                             w2v_embd: label_w2v})
#                     writer.add_summary(s, epoch * train_batches_per_epoch + step)
#
#     #Validation start
#             val_loss = 0.
#             val_acc = 0.
#             val_count = 0
#             if (epoch+1) % validation_epoch == 0:
#                 print("{} Start validation".format(datetime.now()))
#                 sess.run(validation_init_op)
#                 for _ in range(val_batches_per_epoch):
#                     img_batch, label_batch = sess.run(next_batch)
#                     y_label = sess.run(tf.argmax(label_batch, 1))
#
#                     t_label_batch = np.zeros(shape=(batch_size, w2v_dim))
#                     t_j_batch = np.zeros(shape=(batch_size, num_classes, w2v_dim))
#                     for i in range(batch_size):
#                         for j in range(num_classes):
#                             if j == y_label[i]:
#                                 t_label_batch[i] = get_normalize_w2v(j)
#                             else:
#                                 t_j_batch[i][j] = get_normalize_w2v(j)
#                     loss, acc = sess.run([margin_loss, accuracy], feed_dict={x: img_batch,
#                                                                              y: label_batch,
#                                                                              keep_prob: 1.,
#                                                                              t_label: t_label_batch,
#                                                                              t_j: t_j_batch,
#                                                                              w2v_embd: label_w2v})
#
#                     val_loss += loss
#                     val_acc += acc
#                     val_count += 1
#                 print("Step " + str(epoch * train_batches_per_epoch + step) + ", Minibatch Loss= " + \
#                       "{:.4f}".format(val_loss/val_count) + ", val Accuracy= " + \
#                       "{:.3f}".format(val_acc/val_count))
#
#             if (epoch + 1) % save_epoch == 0:
#                 print("{} Saving checkpoint of model...".format(datetime.now()))
#
#                 # save checkpoint of the model
#                 checkpoint_name = os.path.join(checkpoint_path,
#                                                'model_epoch' + str(epoch + 1) + '.ckpt')
#                 save_path = saver.save(sess, checkpoint_name)
#
#                 print("{} Model checkpoint saved at {}".format(datetime.now(),
#                                                                checkpoint_name))
#
#     #total_score /= test_count
#         total_loss /= train_count
#         print("{} Training Loss = {:.4f}".format(datetime.now(),
#                                                        total_loss))
#         print("{} Saving checkpoint of model...".format(datetime.now()))
#
#         # save checkpoint of the model
#         checkpoint_name = os.path.join(checkpoint_path,
#                                        'model_epoch' + str(epoch + 1) + '.ckpt')
#         save_path = saver.save(sess, checkpoint_name)
#
#         print("{} Model checkpoint saved at {}".format(datetime.now(),
#                                                        checkpoint_name))
#
#     print("{} Start Test".format(datetime.now()))
#     sess.run(testing_init_op)
#     test_loss = 0.
#     test_acc = 0.
#     test_count = 0
#     for _ in range(test_batches_per_epoch):
#         img_batch, label_batch = sess.run(next_batch)
#         y_label = sess.run(tf.argmax(label_batch, 1))
#
#         t_label_batch = np.zeros(shape=(batch_size, w2v_dim))
#         t_j_batch = np.zeros(shape=(batch_size, num_classes, w2v_dim))
#         for i in range(batch_size):
#             for j in range(num_classes):
#                 if j == y_label[i]:
#                     t_label_batch[i] = get_normalize_w2v(j)
#                 else:
#                     t_j_batch[i][j] = get_normalize_w2v(j)
#         loss, acc = sess.run([margin_loss, accuracy], feed_dict={x: img_batch,
#                                                                  y: label_batch,
#                                                                  keep_prob: 1.,
#                                                                  t_label: t_label_batch,
#                                                                  t_j: t_j_batch,
#                                                                  w2v_embd: label_w2v})
#
#         print("Step " + str(_) + ", Test Loss= " + \
#               "{:.4f}".format(loss) + ", Test Accuracy= " + \
#               "{:.3f}".format(acc))
#         test_loss += loss
#         test_acc += acc
#         test_count += 1
#     print("test Loss= " + \
#           "{:.4f}".format(test_loss/test_count) + ", test Accuracy= " + \
#           "{:.3f}".format(test_acc/test_count))
#
#