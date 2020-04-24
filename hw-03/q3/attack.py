#!/usr/bin/env python

# vim: ts=4 sw=4 sts=4 expandtab

import numpy as np
import tensorflow as tf
from common import config
from itertools import product
import cv2

mean, std = config.mean, config.std

class Attack():

    def __init__(self, model, batchsize, **kwargs):
        self.batchsize = batchsize
        self.model = model  # pretrained vgg model used as classifier
    
    '''Build computation graph for generating adversarial examples'''
    def generate_graph(self, pre_noise, x, gt, target = None, **kwargs):
        noise = 10 * tf.tanh(pre_noise) 
        x_noise = x + noise                 ## add perturbation and get adversarial examples
        # x_augment = tf.contrib.image.translate(x_noise, [10, 10])
        x_noise = self.augment(x_noise, 'rotation', 'pepper', 'blurring')
        x_clip = tf.clip_by_value(x_noise, 0, 255) 
        x_round = x_clip + tf.stop_gradient(x_clip // 1 - x_clip) ##skip computing gradient wrt to rounded results(x_round) and only calculate the gradient wrt to x_clip 
        x_norm = (x_round - mean)/(std + 1e-7)          ## normalize the image input for the classfier
        logits = self.model.build(x_norm)               
        preds = tf.nn.softmax(logits)
        gt_one_hot = tf.one_hot(gt, config.nr_class)
        if target != None:
            target_one_hot = tf.one_hot(target, config.nr_class)
        else:
            target_one_hot = tf.one_hot(gt, config.nr_class)

        loss = tf.losses.softmax_cross_entropy(target_one_hot, logits) * (-1)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                             tf.cast(tf.argmax(gt_one_hot, 1), dtype = tf.int32)), tf.float32))
        return  acc, loss, x_round
    
    def augment(self, image, *argv):
        for augment in argv:
            if augment == 'rotation':
                image = tf.image.flip_up_down(image= image)
            elif augment == 'pepper':
                filter_shape = (config.image_shape[0], config.image_shape[1], 3)
                rand_filter = np.random.sample(filter_shape);
                for i, j, k in product(range(filter_shape[0]), range(filter_shape[1]), range(filter_shape[2])):
                    if rand_filter[i, j, k] < 0.2:
                        rand_filter[i, j, k] = -255
                    elif rand_filter[i, j, k] > 0.8:
                        rand_filter[i, j, k] = 255
                    else:
                        rand_filter[i, j, k] = 0
                rand_filter = tf.constant(rand_filter, dtype= tf.float32, shape= filter_shape)
                image = tf.add(image, rand_filter)
            elif augment == 'blurring':
                kernel_size = 4
                gamma = 1
                kernel = np.matmul(cv2.getGaussianKernel(kernel_size, gamma), np.transpose(cv2.getGaussianKernel(kernel_size, gamma)))
                gauss_filter = tf.constant(kernel, dtype= tf.float32, shape= [kernel_size, kernel_size, 3, 3])
                image = tf.nn.conv2d(input= image, filter= gauss_filter, strides= [1, 1, 1, 1], padding= 'SAME', data_format= 'NHWC')
        return image

    '''Build a graph for evaluating the classification result of adversarial examples'''
    def evaluate(self, x, gt, **kwargs): 
        x = (x - mean)/(std + 1e-7)
        logits = self.model.build(x)
        preds = tf.nn.softmax(logits)
        gt_one_hot = tf.one_hot(gt, config.nr_class)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                             tf.cast(tf.argmax(gt_one_hot, 1), dtype = tf.int32)), tf.float32))


        return acc
