import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import os.path
import time

data_folder = "./test_image/"
train_images = []
train_labels = []
test_images = []
test_labels = []

def resize_train_image():
    for i in range(0, 100):
        filename = data_folder + 'test' + str(i) + '.jpg'
        if os.path.exists(filename):
            pic = pil.open(filename)
            new_pic = pic.resize((32, 32))
            np_array = np.array(new_pic)
            train_images.append(np_array)
            train_labels.append(0)
        else:
            break    
#cifar_mnist = datasets.cifar10
#(train_images, train_labels), (test_images, test_labels) = cifar_mnist.load_data()
#print(train_images.dtype)

resize_train_image()
batch_size = 64
num_classes = 10
epochs = 35

train_images = np.array(train_images, dtype = np.float32)
train_images = train_images / 255

test_images = np.array(test_images, dtype = np.float32)
test_images = test_images / 255

train_labels = utils.to_categorical(train_labels, num_classes)
test_labels = utils.to_categorical(test_labels, num_classes)

print(len(train_images))