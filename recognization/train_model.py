
# coding: utf-8

# In[1]:


# coding: utf-8

# In[1]:


import changeubyte
import helper
import tensorflow as tf
import numpy as np
import struct
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from PIL import Image
import os

def load_minst():
    train_images_idx3_ubyte_file = './data/emnist-letters-train-images-idx3-ubyte'
    train_labels_idx1_ubyte_file = './data/emnist-letters-train-labels-idx1-ubyte'
    test_images_idx3_ubyte_file = './data/emnist-letters-test-images-idx3-ubyte'
    test_labels_idx1_ubyte_file = './data/emnist-letters-test-labels-idx1-ubyte'

    train_images = changeubyte.decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = changeubyte.decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_images = changeubyte.decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels = changeubyte.decode_idx1_ubyte(test_labels_idx1_ubyte_file)
 
    etrain = train_images/255
    etrainl = to_categorical(train_labels)

    etest = test_images/255
    etestl = to_categorical(test_labels)
    
    return etrain,etrainl,etest,etestl

def weight_variable(shape,name1):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name1)

def bias_variable(shape,name1):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name1)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def train_model(etrain,etrainl,etest,etestl):
    
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 27])
    
    W_conv1 = weight_variable([5, 5, 1, 32],name1='W1')
    b_conv1 = bias_variable([32],name1='b1')
    x_image = tf.reshape(x, [-1,28,28,1])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64],name1='W2')
    b_conv2 = bias_variable([64],name1='b2')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024],name1='W3')
    b_fc1 = bias_variable([1024],name1='b3')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 27],name1='W4')
    b_fc2 = bias_variable([27],name1='b4')
    
    #variables_dict = {'W1': W_conv1, 'b1': b_conv1, 'W2': W_conv2, 'b2': b_conv2, 'W3': W_fc1, 'b3': b_fc1, 'W4':W_fc2, 'b4':b_fc2}
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    flag=0
    saver_path="sorry,the accuracy is to low,please try again1"
    
    for i in range(20000):
        batch_x,batch_y=helper.next_batch(etrain,etrainl,100)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    saver.save(sess, "./model2/cnnnetwork_model.ckpt")
    print('Ssve suceffully')
    test_accuracy=accuracy.eval(feed_dict={ x: etest, y_: etestl, keep_prob: 1.0})
    sess.close()
    return test_accuracy

etrain,etrainl,etest,etestl=load_minst()
result=train_model(etrain,etrainl,etest,etestl)
print(result)

