import tensorflow as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import utils
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import datasets
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import os.path
import time
import cv2
import json
from tensorflow.compat.v1.keras import backend as K

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()
#test이미지가 저장되어 있는 폴더
data_folder = "./test_image"
img_size_file = "img_size.txt"
MININUM_TEST_NUM = 10

#훈련된 weight를 저장한 경로와 파일 이름
checkpoint_path = "model"
checkpoint_dir = "./checkpoints/"
checkpoint_num = 13

#이미지 정보 저장
width = 0
height = 0

class_names = [
    'blackbean', 'herbsalt', 'homerun', 'lion', 'narangd', 'rice', 'sixopening', 'skippy', 'BlackCap', 'CanBeer', 'doritos',
    'Glasses', 'lighter', 'mountaindew', 'pepsi', 'Spoon',  'tobacco', 'WhiteCap', 'note'
]

#class의 길이를 정한다.
num_classes = len(class_names)

class_prices = [
    1000, 800, 1500, 6000, 1000, 1500, 800, 800, 25000, 2000, 1500, 50000, 4000, 1000, 1000, 1000, 1500, 30000, 2000
]

class_counts = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

#test이미지의 저장된 이름
test_names = [
    'testset',
]

#load된 model를 저장할 곳
test_model = []
#test이미지를 저장하는 곳
test_images = []

#output을 내기 위한 변수들
result_labels = []
result_labels_set = []
final_result_labels = []
final_result_count = []

#test이미지 파일을 load 시킨다
def load_img_size():
    global width
    global height
    #width, height를 load 시킨다.
    fp = open(img_size_file, "r")
    size = fp.readline()
    if size == None:
        print('Image Size Error')
        return
    width, height = size.split(' ')
    width = int(width)
    height = int(height)
    fp.close()

    return (width, height)

def load_test_img(folder_num):
    global test_images
    global test_names

    test_images = []

    (width, height) = load_img_size()
    for test_folders in test_names:
        for image_seq in range(MININUM_TEST_NUM + 1):
            filename = test_folders + str(folder_num) + '/' + test_folders + str(image_seq) + '.jpg'
            if os.path.exists(filename) == False:
                continue
            img = cv2.imread(filename)
            img = cv2.resize(img, dsize = (height, width), interpolation = cv2.INTER_AREA)
            img = list(img)
            test_images.append(img)
    
    test_images = np.array(test_images)

def load_model():
    global test_model
    global sess
    
    K.set_session(sess)
    for idx in range(checkpoint_num):
        md = Model(str(idx))
        md.make_model()
        md.model_compile()
        md.load_weights()
        test_model.append(md)

def output_result():
    global test_model
    global class_names
    global result_labels
    global result_labels_set
    global test_images

    dict_for_result = {}
    qurom = len(test_model) * 0.5

    for img in test_images:
        test_img = []
        test_img.append(img)
        test_img = np.array(test_img) 
        #여러개의 품목이 있을 수 있으므로 DICTIONARY 형태로 저장해 놓는다.
        #과자 : 1개, 음료수 : 2개.... 이런식으로 저장한다. (한 test 이미지 당)
        predictions = np.zeros(1 * num_classes).reshape(1, num_classes)

        for idx, md in enumerate(test_model):
            p = md.predictions(test_img)
            index = np.argmax(p)
            if p[0][index] < .5:
                break
            predictions += p

        index = np.argmax(predictions)
        if predictions[0][index] < qurom:
            continue
        #물건 이름을 우선 print해서 출력한다.
        thing_name = class_names[index]
        print(thing_name)
        #print(thing_name)
        
        #우선 이미 품목이 있는지 검사한다.
        if dict_for_result.get(index):
            dict_for_result[index] += 1
        else:
            dict_for_result[index] = 1
        
        result_labels_set.append(index)
    
    #최종 label에 test당 dictionary를 붙여넣는다.
    result_labels.append(dict_for_result)
    #중복되는 물품을 없애고, 투표를 하기위해 중복물품을 없앤다.
    result_labels_set = set(result_labels_set)
    result_labels_set = list(result_labels_set)

#가격을 계산하기 위한 전제함수 이다.
def prev_calculate_price():
    global result_labels
    global result_labels_set
    global final_result_labels
    global final_result_count

    quorm = int(MININUM_TEST_NUM / 2)

    #item을 하나씩 꺼내고 test마다 돌면서 투표수를 센다
    for item in result_labels_set:
        vote = 0
        #중복 물품이 있을 경우, 물품의 개수를 세기 위해 존재한다. 2개 : 1, 1개 : 2..... 이런식으로
        dict_for_item_count = {}
        for test in result_labels:
            if test.get(item):
                vote += 1
                #개수를 저장해 놓는다.
                if dict_for_item_count.get(test[item]):
                    dict_for_item_count[test[item]] += 1
                else:
                    dict_for_item_count[test[item]] = 0
        if quorm >= vote:
            final_result_labels.append(item)
            sort_item_count = sorted(dict_for_item_count.items(), reverse = True, key = lambda item : item[1])
            for key, _ in sort_item_count:
                final_result_count.append(key)
                break
    
    #후의 reuse를 위해 초기화 해놓는다.
    result_labels = []
    result_labels_set = []

def calculate_price(mydb):
    global final_result_labels
    global final_result_count

    result_dict = {}

    sum = 0
    for index, count in zip(final_result_labels, final_result_count):
        sum += (class_prices[index] * count)
        #json file로 dump하기 위한 코드
        #미래에 server에서 사용한다.
        result_dict[class_names[index]] = count
    
    if apply_db(mydb, result_dict) == False:
        print('Cannot Apply Because Product count is not enough')
        for key, value in result_dict.items():
            result_dict[key] = -1
    
    with open("product_file.json", "w") as json_file:
        json.dump(result_dict, json_file)
    
    print('최종 가격 : ' + str(sum))

    #후의 reuse를 위해 초기화 해놓는다.
    final_result_count = []
    final_result_labels = []

class Model():
    global checkpoint_dir
    global checkpoint_path
    global num_classes
    global sess
    global graph

    def __init__(self, name):
        self.name = name
        self.num_classes = num_classes
    
    def make_model(self, drop_out = .25):
        self.model = keras.Sequential([
            Conv2D(16, kernel_size = (3, 3), padding = 'same', input_shape = (width, height, 3), activation = tf.nn.relu),
            MaxPooling2D(pool_size = (2, 2)),
            Dropout(drop_out),

            Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = tf.nn.relu),
            MaxPooling2D(pool_size = (2, 2)),
            Dropout(drop_out),

            Flatten(),
            Dense(64, activation = tf.nn.relu),
            Dropout(drop_out),
            Dense(self.num_classes, activation = tf.nn.softmax)    
        ])
    
    def model_compile(self):
        #모델의 shape를 그 shape대로 compile한다.
        if self.model == None:
            print('There is no model')
            return
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy']
        )
    
    def load_weights(self):
        self.model.load_weights(checkpoint_dir + checkpoint_path + str(self.name))

    def predictions(self, test_image):
        K.set_session(sess)
        return self.model.predict(test_image)

#여기서 result_dict를 통해 db에 결과를 적용시킨다.
def apply_db(mydb, result_dict):
    #우선 count가 0으로 안떨어지는지 확인한다.
    for key, value in result_dict.items():
        for idx, product_name in enumerate(class_names):
            if product_name == key:
                if class_counts[idx] - value < 0:
                    return False

    #적용시킨다.
    for key, value in result_dict.items():
        for idx, product_name in enumerate(class_names):
            if product_name == key:
                mydb.apply_result_db(key, class_counts[idx] - value)
    return True
'''
if __name__ == "__main__":
    load_img_size()
    load_test_img(0)
    load_model()
    output_result()
'''


