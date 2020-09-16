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

'''
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
'''
cifar_mnist = datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_mnist.load_data()

class_names = [
    'Airplane', 'Car', 'Birs', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship','Truck'
]
'''
#for train_image
def draw_train_image():
    plt.figure(figsize = (10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap = plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()
#for test_image
def draw_test_image():
    plt.figure(figsize = (10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap = plt.cm.binary)
        plt.xlabel(class_names[test_labels[i][0]])
    plt.show()
'''
class Model():
    global num_classes
    global epochs

    def __init__(self, name):
        self.name = name
        self.num_classes = num_classes

    def make_model(self, train_images, drop_out = 0.25):
        self.model = keras.Sequential([
            Conv2D(32, kernel_size = (3, 3), padding = 'same', input_shape = train_images.shape[1:], activation = tf.nn.relu),
            MaxPooling2D(pool_size = (2, 2)),
            Dropout(drop_out),

            Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu),
            MaxPooling2D(pool_size = (2, 2)),
            Dropout(drop_out),

            Flatten(),
            Dense(64, activation = tf.nn.relu),
            Dropout(drop_out),
            Dense(self.num_classes, activation = tf.nn.softmax)
        ])
    
    def model_compile(self):
        if self.model == None:
            print('There is no model')
            return
        
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy']
        )
    
    def model_fit(self, train_images, train_labels):
        self.early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)

        self.history = self.model.fit(
            train_images,
            train_labels,
            epochs = epochs,
            validation_data = (train_images, train_labels),
            shuffle = True,
            callbacks = [self.early_stopping]
        )

    def predictions(self, test_images):
        return self.model.predict(test_images)
    
    def description(self):
        if self.model == None:
            print('There is no model')
            return
        print(self.model.summary())

batch_size = 1000
num_classes = 10
epochs = 35
total_image_num = int(len(train_images) / batch_size)
model = []

train_images = np.array(train_images, dtype = np.float32)
train_images = train_images / 255

test_images = np.array(test_images, dtype = np.float32)
test_images = test_images / 255

train_labels = utils.to_categorical(train_labels, num_classes)
test_labels = utils.to_categorical(test_labels, num_classes)


for i in range(0, total_image_num):
    batch_model = Model(str(i))
    split_train_image = train_images[i * batch_size : (i + 1) * batch_size, :]
    split_train_labels = train_labels[i * batch_size : (i + 1) * batch_size, :]
    batch_model.make_model(split_train_image)
    batch_model.model_compile()
    batch_model.model_fit(split_train_image, split_train_labels)
    model.append(batch_model)

test_size = len(test_images)
predictions = np.zeros(test_size * num_classes).reshape(test_size, num_classes)

for i in range(0, total_image_num):
    p = model[i].predictions(test_images)
    predictions += p

sess = tf.compat.v1.Session()
ensemble_correct_prediction = tf.compat.v1.equal(
    tf.compat.v1.argmax(predictions, 1), tf.compat.v1.argmax(train_labels, 1))
ensemble_accuracy = tf.compat.v1.reduce_mean(
    tf.compat.v1.cast(ensemble_correct_prediction, tf.compat.v1.float32))
print('Ensemble accuracy : ', sess.run(ensemble_accuracy))




