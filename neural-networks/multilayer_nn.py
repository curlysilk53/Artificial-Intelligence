from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.models import Sequential
from keras.layers import Input, Dense # the two types of neural network layer we will be using
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras import layers
from PIL import Image
import numpy as np
import os
import re
import sys
from random import shuffle

TRAIN_DIR = "./train"

def load_train_data():
    train_data = []
    for img in os.listdir(TRAIN_DIR):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        train_data.append([np.array(img), label])
    return train_data

TEST_DIR = './test'
def load_test_data():
    test_data = []
    for img in os.listdir(TEST_DIR):
        label = label_img(img)
        path = os.path.join(TEST_DIR, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        test_data.append([np.array(img), label])
    shuffle(test_data)
    return test_data

def label_img(img_path):
    match = re.match("(\w+)_\d+\.png", img_path)
    alpha = match.group(1)
    if alpha == "u":
        return np.array([1, 0, 0])
    elif alpha == "e":
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])

IMG_SIZE = 32

train_data = load_train_data()
test_data = load_test_data()

trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array([i[1] for i in train_data])

testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
testLabels = np.array([i[1] for i in test_data])

num_train = 295
num_test = 127
print(len(os.listdir(TRAIN_DIR)))
print(len(os.listdir(TEST_DIR)))

trainImages = trainImages.reshape(num_train, IMG_SIZE * IMG_SIZE)
testImages = testImages.reshape(num_test, IMG_SIZE * IMG_SIZE) 

trainImages = trainImages.astype('float32') 
testImages = testImages.astype('float32')
trainImages /= 255 
testImages /= 255 


inp = Input(shape=(IMG_SIZE * IMG_SIZE,))
hidden_1 = Dense(64, activation='relu')(inp) 
hidden_2 = Dense(64, activation='relu')(hidden_1) 
# hidden_3 = Dense(32, activation='relu')(hidden_2) # Second hidden ReLU layer 32
out = Dense(3, activation='softmax')(hidden_2)

model = Model(input=inp, output=out) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # reporting the accuracy

model.fit(trainImages, trainLabels, batch_size=2, epochs=150,verbose=1)
loss, acc = model.evaluate(testImages, testLabels, verbose=1)
print(acc * 100)
