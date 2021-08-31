# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# 사용하시기전 https://drive.google.com/file/d/1r4mvtfiCotcdmnrN6ieCZF0MbYKjPkuN/view?usp=sharing의 꽃데이터셋 zip
# 구글드라이브 datasets 폴더에 넣어주세요
# Ignore  the warnings
import warnings

import joblib
import matplotlib

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
# data visualisation and manipulation
import numpy as np
import zipfile
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

# configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder

# preprocess.
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# specifically for cnn
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
import random as rn
# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import sys
sys.setrecursionlimit(2000)
import cv2
from tqdm import tqdm
import os

# # #구글드라이브에있는 flowers.zip파일 압축해제
# import gdown
#
# google_path = 'https://drive.google.com/uc?id='
# file_id = '1r4mvtfiCotcdmnrN6ieCZF0MbYKjPkuN'
# output_name = 'flowers.zip'
# gdown.download(google_path+file_id,output_name,quiet=False)
# #
# from google.colab import drive
# drive.mount('/content/drive')
#
# currentP=os.getcwd()
# print(currentP)
# os.listdir()
# os.chdir("/content/drive/MyDrive/datasets")
# os.listdir()
#
# if not os.path.isdir('./flowers2'):
#     print('처음 실행합니다.')
#     with zipfile.ZipFile('./flowers2.zip') as zf:
#         zf.extractall(path='./flowers2')
#     print('압축을 풀었습니다.')
# else :
#     print('이미 폴더가 존재합니다.')
# =========================================================
base_dir = '\User\ukiii\PycharmProjects\pythonProject3\ml'
os.chdir(base_dir)

X = []
Z = []
# FLOWER_장미_푸에고_DIR = './flowers2/장미_푸에고/'
# FLOWER_국화_백선_DIR = './flowers2/국화_백선/'
# FLOWER_리시안사스_졸리핑크_DIR = './flowers2/리시안사스-졸리핑크/'
# FLOWER_거베라_거베라_DIR = './flowers2/거베라_거베라/'
# FLOWER_수국_그린_DIR = './flowers2/수국_그린/'
# 클래스추가시 추가코드작성
FLOWER_lily_siberia_DIR = './flowers2/lily_siberia/'
FLOWER_sunflower_sunflower_DIR = './flowers2/sunflower_sunflower/'
FLOWER_gentianascabra_gentianascabra_DIR = './flowers2/gentianascabra_gentianascabra/'
FLOWER_gypsophilaist_overtime_DIR = './flowers2/gypsophilaist_overtime/'
FLOWER_carnation_redcarnation_DIR = './flowers2/carnation_redcarnation/'


def assign_label(img, flower_type):
    return flower_type


def make_train_data(flower_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, flower_type)
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print('Wrong path:', path)
        else:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(np.array(img))
            Z.append(str(label))


IMG_SIZE = 150

# X,Z=make_train_data('장미_푸에고',FLOWER_장미_푸에고_DIR)
# print(len(X))
# X,Z=make_train_data('국화-백선',FLOWER_국화-백선_DIR)
# print(len(X))
# X,Z=make_train_data('리시안사스_졸리핑크',FLOWER_리시안사스_졸리핑크_DIR)
# print(len(X))
# X,Z=make_train_data('거베라_거베라',FLOWER_거베라_거베라_DIR)
# print(len(X))
# X,Z=make_train_data('수국_그린',FLOWER_수국_그린_DIR)
# print(len(X))


# 클래스별로 진행, 클래스추가시 추가코드작성할것


make_train_data('백합_시베리아', FLOWER_lily_siberia_DIR)
print(len(X))
make_train_data('해바라기_해바라기', FLOWER_sunflower_sunflower_DIR)
print(len(X))
make_train_data('용담_용담', FLOWER_gentianascabra_gentianascabra_DIR)
print(len(X))
make_train_data('안개_오버타임', FLOWER_gypsophilaist_overtime_DIR)
print(len(X))
make_train_data('카네이션_빨간카네이션', FLOWER_carnation_redcarnation_DIR)
print(len(X))

# 전처리
X = np.array(X)
X = X / 255
le = LabelEncoder()
Y = le.fit_transform(Z)
Y = to_categorical(Y, 5)

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)


from keras.applications.xception import Xception

keras.backend.clear_session()

model = Xception(weights='imagenet',
                 include_top=False,
                 input_shape=(IMG_SIZE, IMG_SIZE, 3))
new_output = GlobalAveragePooling2D()(model.output)
new_output = Dropout(0.5)(new_output)
new_output = Dense(5, activation='softmax')(new_output)

model = keras.models.Model(model.inputs, new_output)

batch_size = 32
epochs = 10

from keras.callbacks import ReduceLROnPlateau

red_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=1, verbose=1, factor=0.1)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs, validation_data=(x_test, y_test),
                    verbose=1, callbacks=[red_lr], steps_per_epoch=x_train.shape[0] // batch_size)
model.save("model.h5")
print("Saved model to disk")