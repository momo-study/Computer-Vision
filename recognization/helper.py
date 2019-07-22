
# coding: utf-8

# In[2]:

from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

def next_batch(image,label,batch_len):
    index=[i for i in range(label.shape[0])]
    np.random.shuffle(index)
    
    img_batch= np.zeros(shape=(batch_len,784))
    lab_batch= np.zeros(shape=(batch_len,27))
    for i in range(0,batch_len):
        img_batch[i]=image[index[i]]
        lab_batch[i]=label[index[i]]
        
    return img_batch,lab_batch

def imgprocess(img1):
    ima = img1.transpose(Image.ROTATE_90)
    img = ima.resize((28,28))
    img = img.convert('L')
    temp = list(img.getdata())
    data=np.array([(255-x)*1.0/255.0 for x in temp])
    return data

def testimgprocess(img1):
    ima = img1.transpose(Image.ROTATE_90)
    ima = ima.transpose(Image.FLIP_TOP_BOTTOM)
    ima = ima.convert('L')
    ima=ima.resize((28,28))
    #temp = list(img.getdata())
    #data=np.array([(255-x)*1.0/255.0 for x in temp])
    data=np.array(ima)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if data[i][j]!=0:
                data[i][j]=255
    data=data/255.0
    #print(data.shape)
    return data