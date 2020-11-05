import tensorflow as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import utils
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import datasets
from tensorflow.compat.v1.keras.callbacks import EarlyStopping
from tensorflow.compat.v1.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
import os.path
import time
import cv2
from tensorflow.compat.v1.keras import backend as K
import copy
#import cash_db

gpus = tf.compat.v1.config.experimental.list_physical_devices('GPU')
tf.compat.v1.config.experimental.set_memory_growth(gpus[0], True)

tf.compat.v1.disable_eager_execution()

#데이터 폴더의 이미지를 LOADING 한다.
train_data_folder = "./train_image/"
test_data_folder = "./test_image/"
MINIMUM_TRAIN_IMAGE_NUM = 300
MINIMUM_TEST_IMAGE_NUM = 1

#훈련중인 모델을 저장할 경로와 파일 이름
checkpoint_path = 'model'
checkpoint_dir = './checkpoints/'

train_images = []
train_labels = []
test_images = []
test_labels = []

result_labels = []
result_labels_set = []
final_result_labels = []
final_result_count = []

dict_for_result = {}

#우리가 분류해야될 물품의 목록을 모아놓는다.
'''
class_names = [
    'blackbean', 'herbsalt', 'homerun', 'lion', 'narangd', 'rice', 'sixopening', 'skippy', 'BlackCap', 'CanBeer', 'doritos',
     'lighter', 'mountaindew', 'pepsi', 'Spoon',  'tobacco', 'WhiteCap', 'note'
]

class_prices = [
    20000, 50000, 35000, 20000, 10000, 170000, 50000, 40000, 30000, 10000, 11200, 10000, 13300, 12000, 13000, 14000, 20000, 42000, 24000, 23000
]

class_counts = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
'''

class_names = [
    'bagepant', 'black', 'blackpant', 'bluepant', 'brown', 'brownpant', 'green', 'greenpant', 'kokky', 'kokkypant',
    'mint', 'nam', 'orange', 'orangepant', 'pink', 'purplepant', 'red', 'redpant', 'yellow', 'yellowpant']

class_prices = [
    20000, 50000, 35000, 20000, 10000, 170000, 50000, 40000, 30000, 10000, 11200, 10000, 13300, 12000, 13000, 14000, 20000, 42000, 24000, 23000
]

class_counts = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

#test이미지의 저장된 이름
test_names = [
    'testset',
]

dict_class_names = {}

batch_size = 1000
num_classes = len(class_names)
epochs = 10
#model = []
model = None
#최종 accuracy가 가장 높은 model을 저장해 놓는다.
final_model = None
#각 훈련 집합마다 accuracy를 측정하여 가장 높은 정확도를 가지는 모델을 가지고 있는다.
global_accuracy = 0.0
data_set_len = 0

#db에 있는 물품을 끌어올린 후에 다시 훈련시키기 위해
'''
mydb = cash_db.MySql(user = 'root', password = 'root', db_name = 'smart_cashier')
mydb.connect()
result = mydb.select_all()
for data in result:
    class_names.append(data[0])
    class_prices.append(data[1])
#여기서 끝
mydb.close_db()
'''
#class_names의 list를 0, 1, 2.... 숫자 label로 바꾼다
def list_to_dictionary():
    global dict_class_names
    for i in range(0, num_classes):
        dict_class_names[class_names[i]] = i

#우리가 모아놓은 폴더에서 image와 label를 load한다.
#데이터에 맞게 label를 load해야 한다. ex) 의자 : 0, 과자 : 1, 핸드폰 : 2....
def load_data():
    global train_images
    global train_labels
    global test_images
    global test_labels
    global final_model

    #label를 위한 키 값 정리
    list_to_dictionary() 
    #훈련 데이터 평균 size 하기
    #훈련 데이터 및 label 정의하기
    
    #(width, height) = data_img_size()
    #이미지의 사이즈를 txt로써 저장한다.
    #fp = open("img_size.txt", "w")
    #fp.write(str(width) + ' ' + str(height))
    #fp.close()
    (width, height) = (128, 128)
    train_data_img(width, height)
    print('------------------ 총 데이터 개수 : {} -------------------------'.format(len(train_images)))   
    test_data_img(width, height)
    #훈련 및 테스티 데이터를 섞어놓는다.
    data_shuffle_train()
    data_shuffle_test()

    #범주 안에 들어가게 하기 위해
    train_images = train_images / 255
    test_images = test_images / 255
    
    #cifar_mnist = datasets.cifar10
    #(train_images, train_labels), (test_images, test_labels) = cifar_mnist.load_data()
    #print(test_images.shape)
    '''
    train_images = np.array(train_images, dtype = np.float32)
    train_images = train_images / 255

    test_images = np.array(test_images, dtype = np.float32)
    test_images = test_images / 255

    train_labels = utils.to_categorical(train_labels, num_classes)
    test_labels = utils.to_categorical(test_labels, num_classes)

    print('-------------------------------------------------')
    print(train_labels.shape)
    '''
    print('\n\n\nTraining Start\n\n\n')
    #one-hot 인코딩을 위해 범주형 변수를 변환시킨다
    train_labels = utils.to_categorical(train_labels, num_classes)
    #훈련 시작
    split_data_and_train_validate(train_images, train_labels)
    #훈련된 모델을 load시킨다.   

#image 의 사이즈를 구한다.
def data_img_size():
    global MINIMUM_TRAIN_IMAGE_NUM
    width = 0
    height = 0
    total_num = 0

    for class_folder in class_names:
        for image_seq in range(MINIMUM_TRAIN_IMAGE_NUM):
            filename = data_folder + class_folder + '/' + class_folder + str(image_seq) + '.jpg' 
            if os.path.exists(filename) == False:
                continue
            img = cv2.imread(filename)
            img = list(img)
            width += len(img)
            height += len(img[0])
            total_num += 1
    
    width = width / total_num
    height = height / total_num
    width = int(width)
    height = int(height)
    return (width, height)

def test_data_img(width, height):
    global test_images
    global test_labels
    global MINIMUM_TEST_NUM

    for class_folder in class_names:
        #이제는 최소의 사진 개수를 위해 FOR문을 돌린다.
        for image_seq in range(MINIMUM_TEST_IMAGE_NUM):
            filename = test_data_folder + class_folder + '/' + class_folder + str(image_seq) + '.jpg' 
            if os.path.exists(filename) == False:
                continue
            img = cv2.imread(filename)
            img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
            img = list(img)
            test_images.append(img)
            tmp = []
            tmp.append(dict_class_names[class_folder])
            test_labels.append(tmp)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

#train data를 넣는다.
#차원을 낮추기 위해 크기 128x128로 고정
def train_data_img(width, height):
    global train_images
    global train_labels
    global MINIMUM_TRAIN_IMAGE_NUM

    for class_folder in class_names:
        for image_seq in range(MINIMUM_TRAIN_IMAGE_NUM):
            filename = train_data_folder + class_folder + '/' + class_folder + str(image_seq) + '.jpg' 
            if os.path.exists(filename) == False:
                continue
            img = cv2.imread(filename)
            img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
            img = list(img)
            train_images.append(img)
            tmp = []
            tmp.append(dict_class_names[class_folder])
            train_labels.append(tmp)
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

#순서대로 데이터 집합을 나눌때, random하게 데이터가 들어가게 하기 위해 shuffle 한다.
def data_shuffle_train():
    global train_images
    global train_labels

    for_shuffle = np.arange(len(train_images))
    np.random.shuffle(for_shuffle)

    train_images = train_images[for_shuffle]
    train_labels = train_labels[for_shuffle]

def data_shuffle_test():
    global test_images
    global test_labels

    for_shuffle = np.arange(len(test_images))
    np.random.shuffle(for_shuffle)

    test_images = test_images[for_shuffle]
    test_labels = test_labels[for_shuffle]

#batch를 발생시키기 위한 코드
def generate_batches(valid, data_set_len):
    global batch_size
    global train_images
    global train_labels

    for train in range(data_set_len):
        if train == valid:
            continue
        yield (train_images[train * batch_size : (train + 1) * batch_size], train_labels[train * batch_size : (train + 1) * batch_size])

#train_data, validate_data, test_data를 random 하게 나눈다.
#밑에 주석처리 해놓은 함수들은 gpu memory 용량이 클때 사용할 것
'''
def split_data_and_train_validate(train_images, train_labels):
    global batch_size
    global global_accuracy
    global final_model
    global model
    #batch_size로 나눌 수 있는 전체 data_set 개수
    data_set_len = int(len(train_images) / batch_size)
    #valid data set을 놓는다.
    for valid in range(0, data_set_len):
        print("----------------------------------" + str(valid) + "th Start!!!!!!!-----------------------")
        valid_data = train_images[valid * batch_size : (valid + 1) * batch_size]
        valid_label = train_labels[valid * batch_size : (valid + 1) * batch_size]
        train_data_set = []
        train_data_label = []
        #train data set을 모아놓는다.
        for train in range(0, data_set_len):
            if valid == train:
                continue
            train_data_set.append(train_images[train * batch_size : (train + 1) * batch_size])
            train_data_label.append(train_labels[train * batch_size : (train + 1) * batch_size])
        
        #train 함수 호출
        train_model(train_data_set, train_data_label)
        #validate 함수 호출
        accuracy = validate(valid_data, valid_label)
        #모델의 정확도가 높으면, 이 모델을 저장한다.
        print('accuracy : ' + str(accuracy))
        if accuracy >= global_accuracy:
            final_model = copy.deepcopy(model)
            global_accuracy = accuracy
        model.clear()
        break

def train_model(train_data_set, train_data_label):
    global model
    #train data set을 가지고 훈련시킨다.
    for idx, train_data in enumerate(train_data_set):
        batch_model = Model(str(idx))
        batch_model.make_model(train_data)
        batch_model.model_compile()
        batch_model.model_fit(train_data, train_data_label[idx])
        model.append(batch_model)
'''
#여기 이코드는 gpu 메모리 용량이 작을때 사용
def split_data_and_train_validate(train_images, train_labels):
    global batch_size
    global global_accuracy
    global final_model
    global model
    global data_set_len
    #batch_size로 나눌 수 있는 전체 data_set 개수
    data_set_len = int(len(train_images) / batch_size)
    print("---------------" + str(data_set_len) + "개의 배치 시작" + "------------------")
    #valid data set을 놓는다.
    for valid in range(0, data_set_len):
        print("----------------------------------" + str(valid) + "th Start!!!!!!!-----------------------")
        mask = np.ones(len(train_images), dtype = bool)
        mask[valid * batch_size : (valid + 1) * batch_size] = False
        valid_data = train_images[valid * batch_size : (valid + 1) * batch_size]
        valid_label = train_labels[valid * batch_size : (valid + 1) * batch_size]
        train_data_set = train_images[mask]
        train_data_label = train_labels[mask]
        print('------------데이터 분류 완료---------------')
        #train data set을 모아놓는다.
        #모델 만들기, shape도 같이보낸다.
        make_one_model(train_images[0])
        #epoch만큼 같은 데이터를 계속해서 돌림
        '''
        for _ in range(epochs):
            for train in range(0, data_set_len):
                if valid == train:
                    continue
                #여기서는 한번에 하나씩 보내버린다.
                    train_model(train_images[train * batch_size : (train + 1) * batch_size], train_labels[train * batch_size : (train + 1) * batch_size])
        '''

        #gen = generate_batches(valid, data_set_len)
        
        #train 함수 호출
        print('------------모델링 시작---------------')
        train_model(train_data_set, train_data_label)
        #validate 함수 호출
        accuracy = validate(valid_data, valid_label)
        #모델의 정확도가 높으면, 이 모델을 저장한다.
        print('accuracy : ' + str(accuracy))
        if accuracy >= global_accuracy:
            final_model = model
            global_accuracy = accuracy
        model = None

def make_one_model(train_images):
    global model
    
    #0번 모델을 만든다.
    model = Model(str(0))
    model.make_model(train_images)
    model.model_compile()

def train_model(train_data_set, train_data_label):
    global model
    #하나의 모델밖에 없으므로 model만을 이용한다.
    #print(np.array(train_data_set).shape)
    model.model_fit(train_data_set, train_data_label)

def validate(valid_data, valid_label):
    #accuracy를 측정하여 정확도를 비교한다.
    global model
    valid_size = len(valid_label)
    predictions = np.zeros(valid_size * num_classes).reshape(valid_size, num_classes)

    '''
    for idx, md in enumerate(model):
        p = md.predictions(valid_data)
        predictions += p
    '''
    p = model.predictions(valid_data)
    predictions += p

    #gpu를 사용하기 위한 코드
    #backend의 k에서 훈련시켜온 session을 tf에 입력시킨다
    #tf를 통해 앙상블 시킨 모든 모델에 대한 정확도를 측정한다.
    with tf.device("/gpu:0"):
        with tf.compat.v1.Session(graph = K.get_session().graph) as sess:
            ensemble_correct_prediction = tf.compat.v1.equal(
                tf.compat.v1.argmax(predictions, 1), tf.compat.v1.argmax(valid_label, 1))
            ensemble_accuracy = tf.compat.v1.reduce_mean(
                tf.compat.v1.cast(ensemble_correct_prediction, tf.compat.v1.float32))
            return sess.run(ensemble_accuracy)

def output_result(test_images):
    global final_model
    global class_names
    global result_labels
    global result_labels_set
    global dict_for_result
 
    for img in test_images:
        test_img = []
        test_img.append(img)
        test_img = np.array(test_img) 
        #여러개의 품목이 있을 수 있으므로 DICTIONARY 형태로 저장해 놓는다.
        #과자 : 1개, 음료수 : 2개.... 이런식으로 저장한다. (한 test 이미지 당)
        dict_for_result = {}
        predictions = np.zeros(1 * num_classes).reshape(1, num_classes)
        '''
        for idx, md in enumerate(final_model):
            p = md.predictions(test_img)
            predictions += p
        '''
        p = final_model.predictions(test_img)
        predictions += p

        index = np.argmax(predictions)
        #물건 이름을 우선 print해서 출력한다.
        thing_name = class_names[index]
        cv2.imshow("example", np.array(img))
        cv2.waitKey(0)
        print(thing_name)
        
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

def calculate_price():
    global final_result_labels
    global final_result_count

    sum = 0
    for index, count in zip(final_result_labels, final_result_count):
        sum += (class_prices[index] * count)
    
    print('최종 가격 : ' + str(sum))

    #후의 reuse를 위해 초기화 해놓는다.
    final_result_count = []
    final_result_labels = []

#model의 weights를 저장한다.
def model_save():
    global model
    global checkpoint_dir
    global checkpoint_path
    #전체 모델의 가중치를 저장한다.
    '''
    for idx, md in enumerate(final_model):
        md.model.save_weights(checkpoint_dir + checkpoint_path + str(idx))
    '''
    final_model.model.save_weights(checkpoint_dir + checkpoint_path + str(0))
    #model를 reuse하기 위해 초기화 한다.
    #model = []
    model = None
    
class Model():
    global num_classes
    global epochs
    global checkpoint_dir
    global checkpoint_path
    global data_set_len

    def __init__(self, name):
        self.name = name
        self.num_classes = num_classes

    def make_model(self, train_images, drop_out = 0.25):
        #전체 모델의 shpae을 정의 해놓는다.
        self.model = keras.Sequential([
            Conv2D(16, kernel_size = (3, 3), padding = 'same', input_shape = train_images.shape, activation = tf.nn.relu),
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
            print('There is no model')  파 초 빨, 초 빨 파, -> 0~1
            return
        
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy']
        )
    
    def model_fit(self, train_images, train_labels):
        #latest = tf.compat.v1.train.latest_checkpoint('training_1')
        latest = None
        if latest != None:
            #저장해 놨던 가중치를 load시킨다
            self.model.load_weights(latest)
        else:
            #저장해놓은 가중치가 없으면 훈련시킨다
            #callback 함수들 정의
            self.early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
            #self.check_point = tf.compat.v1.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path, save_weights_only = True, verbose = 1, period = 5)
            #gpu로 돌리기 위한 코드
            #코드 실행안되면 tensorflow-gpu 버전에 맞춰 CUDA Toolkit 설치 후 확인
            
            with tf.device('/gpu:0'):
                self.history = self.model.fit(
                    train_images,
                    train_labels,
                    epochs = epochs,
                    validation_data = (train_images, train_labels),
                    shuffle = True,
                    batch_size = 4,
                    #내가 던져준 데이터에서의 batch_size를 나눠서 학습
                    callbacks = [self.early_stopping]
                )
            
            '''
            with tf.device('/gpu:0'):
                self.model.train_on_batch(train_images, train_labels)
            '''
            '''
            with tf.device('/gpu:0'):
                self.history = self.model.fit_generator(
                    generator = gen,
                    steps_per_epoch = data_set_len, 
                    epochs = epochs,
                    callbacks = [self.early_stopping]
                )
            '''

    def predictions(self, test_images):
        #실제 데이터를 입력하여 예측한다.
        return self.model.predict(test_images)
    
    def description(self):
        if self.model == None:
            print('There is no model')
            return
        #모델 전체 shape에 대한 기술을 보여준다.
        print(self.model.summary())

    #이미 훈련된 가중치를 load 시킨다.
    def load_weights(self):
        self.model.load_weights(checkpoint_dir + checkpoint_path + str(self.name))


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    #global global_accuracy
    load_data()
    model_save()
    output_result(test_images)





