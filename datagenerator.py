# Created on Wed May 31 14:48:46 2017
# Modify 18-3-30 9:33
#
# @author: Frederik Kratzert modify by GYxiaOH

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

#IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000, resize=224, mean='',need_c=False,
                 click_fature_1000='', click_fature_200='',click_fature_10=''):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = txt_file
        self.num_classes = num_classes
        self.resize = resize

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)

        #read mean file or set defalut
        if (mean == ''):
            self.mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
        else:
            self.mean = tf.constant(np.load(mean).mean(1).mean(1), tf.float32)

        # # initial shuffling of the file and label lists (together!)
        # if shuffle:
        #     self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        if need_c:

            self.click_feature_1000 = np.loadtxt(click_fature_1000,dtype=float)
            self.click_feature_200 = np.loadtxt(click_fature_200, dtype=float)
            self.click_feature_10 = np.loadtxt(click_fature_10, dtype=float)

            if shuffle:
                self._shuffle_lists_withclick()

            self.click_feature_1000 = convert_to_tensor(self.click_feature_1000,dtype=dtypes.float32)
            self.click_feature_200 = convert_to_tensor(self.click_feature_200, dtype=dtypes.float32)
            self.click_feature_10 = convert_to_tensor(self.click_feature_10, dtype=dtypes.float32)
            data = Dataset.from_tensor_slices((self.img_paths, self.labels,self.click_feature_1000,self.click_feature_200,self.click_feature_10))

            data = data.repeat()

            # distinguish between train/infer. when calling the parsing functions
            data = data.map(self._parse_function_click, num_threads=8,
                            output_buffer_size=99 * batch_size)


            # shuffle the first `buffer_size` elements of the dataset
            if shuffle:
                data = data.shuffle(buffer_size=buffer_size)

            # create a new dataset with batches of images
            data = data.batch(batch_size)

            self.data = data

        else:
            if shuffle:
                self._shuffle_lists()
            # create dataset
            data = Dataset.from_tensor_slices((self.img_paths, self.labels))

            data = data.repeat()

            # distinguish between train/infer. when calling the parsing functions
            if mode == 'training':
                data = data.map(self._parse_function_train, num_threads=8,
                          output_buffer_size=99*batch_size)

            elif mode == 'inference':
                data = data.map(self._parse_function_inference, num_threads=8,
                          output_buffer_size=99*batch_size)

            else:
                raise ValueError("Invalid mode '%s'." % (mode))

            # shuffle the first `buffer_size` elements of the dataset
            if shuffle:
                data = data.shuffle(buffer_size=buffer_size)

            # create a new dataset with batches of images
            data = data.batch(batch_size)


            self.data = data

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip('\n').split(' ')
                self.img_paths.append(' '.join(items[:-1]))
                self.labels.append(int(items[-1]))

    def _shuffle_lists_withclick(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        c_1000 = self.click_feature_1000
        c_200 = self.click_feature_200
        c_10 = self.click_feature_10

        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        self.click_feature_1000 = np.zeros(shape=c_1000.shape,dtype=float)
        self.click_feature_200 = np.zeros(shape=c_200.shape,dtype=float)
        self.click_feature_10 = np.zeros(shape=c_10.shape,dtype=float)
        index = 0
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])
            self.click_feature_1000[index] = c_1000[i]
            self.click_feature_200[index] = c_200[i]
            self.click_feature_10[index] = c_10[i]
            index += 1

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])


    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)


        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [256, 256])
        img_resized = tf.random_crop(img_resized,[self.resize,self.resize,3])
        #orginal method
        #img_resized = tf.image.resize_images(img_decoded, [self.resize, self.resize])
        # img_resized = tf.image.random_flip_left_right(img_resized)
        # img_resized = tf.image.random_flip_up_down(img_resized)
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, self.mean)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [256, 256])
        img_resized = tf.image.resize_image_with_crop_or_pad(img_resized,self.resize,self.resize)
        #img_resized = tf.image.resize_images(img_decoded, [self.resize, self.resize])
        img_centered = tf.subtract(img_resized, self.mean)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

    def _parse_function_click(self,filename,label,clickf_1000,clickf_200,clickf_10):
        """such as train file , add click feature"""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [256, 256])
        img_resized = tf.random_crop(img_resized, [self.resize, self.resize, 3])
        img_centered = tf.subtract(img_resized, self.mean)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        click_feature_1000 = clickf_1000
        click_feature_200 = clickf_200
        click_feature_10 = clickf_10

        return img_bgr, one_hot, click_feature_1000,click_feature_200,click_feature_10
