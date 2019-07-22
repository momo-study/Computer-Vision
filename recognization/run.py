
# coding: utf-8

# In[1]:


import changeubyte
import helper
import tensorflow as tf
import numpy as np
import struct
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os
import glob
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_one_dir(dir_name):
    imgs=[]
    file_names = glob.glob(os.path.join(dir_name, '*.bmp'))
    file_names.sort()
    for file_name in file_names:
        #print('#############################################')
        #print(file_name)
        img=Image.open(file_name)
        imgs.append(img)
    return imgs

def read_some_dir(dir_names):
    dirs=[]
    for dir_name in dir_names:
        imgs=[]
        file_names = glob.glob(os.path.join(dir_name, '*.jpg'))
        for file_name in file_names:
            img=Iamge.open(file_name)
            imgs.append(img)
        dirs.append(imgs)
    return dirs

def process_one_dir(imgs):
    process_imgs=[]
    for img in imgs:
        process_img=helper.testimgprocess(img)
        process_imgs.append(process_img)
    temp_imgs=np.array(process_imgs)
    test_imgs=np.reshape(temp_imgs,(len(process_imgs),784))
    return test_imgs

def process_some_dir(dirs):
    process_dirs=[]
    process_imgs=[]
    dir_num=len(dirs)
    for i in range(dir_num):
        img_num=len(dirs[i])
        for j in range(img_num):
            process_img=helper.imgprocess(dirs[i][j])
            process_imgs.append(process_img)
        process_dirs.append(process_imgs)
    temp_dirs=np.array(process_dirs)
    test_imgs=np.reshape(temp_dirs,(len(process_dirs),len(process_imgs),784))
    return test_imgs    
        
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

def run_one_dir(image):
    
    images=read_one_dir(image)
    test_data=process_one_dir(images)
    
    #print(test_data.shape)
    #test_imgs=np.reshape(test_data[1],(28,28))
    #print(test_imgs)
    #plt.imshow(test_imgs)
    #plt.show()
    index=[]
    letters=[]
    sess = tf.InteractiveSession()
    #init = tf.global_variables_initializer()
    #sess.run(init)
    saver = tf.train.import_meta_graph('./model/cnnnetwork_model.ckpt.meta')  
    saver.restore(sess, './model/cnnnetwork_model.ckpt')
    
    x = tf.placeholder(tf.float32, shape=[None, 784])
    W_conv1=sess.graph.get_tensor_by_name('W1:0')

    b_conv1=sess.graph.get_tensor_by_name('b1:0')
    x_image = tf.reshape(x, [-1,28,28,1])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2=sess.graph.get_tensor_by_name('W2:0')
    b_conv2=sess.graph.get_tensor_by_name('b2:0')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = sess.graph.get_tensor_by_name('W3:0')
    b_fc1 = sess.graph.get_tensor_by_name('b3:0')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = sess.graph.get_tensor_by_name('W4:0')
    b_fc2 = sess.graph.get_tensor_by_name('b4:0')
    #variables_dict = {'W1': W_conv1, 'b1': b_conv1, 'W2': W_conv2, 'b2': b_conv2, 'W3': W_fc1, 'b3': b_fc1, 'W4':W_fc2, 'b4':b_fc2}
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    
    
    W=sess.run(W_fc2, feed_dict={x: test_data, keep_prob: 1.0})
    B=sess.run(b_fc2, feed_dict={x: test_data, keep_prob: 1.0})
    res = sess.run(y_conv, feed_dict={x: test_data, keep_prob: 1.0})
    #print('W1: ',sess.run(W_fc2),'b1: ',sess.run(b_fc2))
    #print(res)
    for i in range(test_data.shape[0]):
        #test_imgs=np.reshape(test_data[i],(28,28))
        #plt.imshow(test_imgs)
        #plt.show()
        index.append(np.argmax(res[i])-1)
        #print(index)
        letters.append(chr(97+index[i]))
        #print('the letters are: ',letters[i])
    sess.close()
    str="".join(letters)
    #print(str)
    with open('./pred_answer.txt', 'a', encoding='utf-8') as f:
        f.write(str)
        f.close()
    return letters

def run_some_dir(dir):
    dirs=read_some_dir(dirs)
    test_dirs=process_some_dir(dirs)
    dir_num=test_dirs.shape[0]
    accuracys=[]
    answers=[]
    f = open("./answer.txt")
    for line in f:
        an_letter=list(line)
        answer.append(an_letter)
    for i in range(dir_num):
        index=[]
        letters=[]
        right=0
        sess = tf.InteractiveSession()
        saver = tf.train.import_meta_graph('./model/cnnnetwork_model.ckpt.meta')  
        saver.restore(sess, './model/cnnnetwork_model.ckpt')
        init = tf.global_variables_initializer()
        sess.run(init)
    
        x = tf.placeholder(tf.float32, shape=[None, 784])
        W_conv1=sess.graph.get_tensor_by_name('W1:0')
        b_conv1=sess.graph.get_tensor_by_name('b1:0')
        x_image = tf.reshape(x, [-1,28,28,1])
    
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        W_conv2=sess.graph.get_tensor_by_name('W2:0')
        b_conv2=sess.graph.get_tensor_by_name('b2:0')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        W_fc1 = sess.graph.get_tensor_by_name('W3:0')
        b_fc1 = sess.graph.get_tensor_by_name('b3:0')
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        W_fc2 = sess.graph.get_tensor_by_name('W4:0')
        b_fc2 = sess.graph.get_tensor_by_name('b4:0')
        #variables_dict = {'W1': W_conv1, 'b1': b_conv1, 'W2': W_conv2, 'b2': b_conv2, 'W3': W_fc1, 'b3': b_fc1, 'W4':W_fc2, 'b4':b_fc2}
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        res = sess.run(y_conv, feed_dict={x: test_data[i], keep_prob: 1.0})
        for j in range(test_data[i].shape[0]):
            index.append(np.argmax(res[0])-1)
            letter=chr(97+index[j])
            letters.append(letter)
            if(letter==answer[i][j]):
                right+=1
        accuracys.append(right/test_data[i].shape[0])
    sess.close()
    return accuracys

result=run_one_dir('test')
print(result)

