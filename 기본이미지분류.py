#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[4]:


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[5]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


# In[6]:


train_images.shape


# In[8]:


len(train_labels)


# In[10]:


train_labels

test_images.shape
# In[12]:


test_images.shape
len(test_labels)


# In[ ]:


#전처리 진행해야함

