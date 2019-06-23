from PIL import Image
import numpy as np
import os
import sys
from random import shuffle
import matplotlib.pyplot as plt
import cv2
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization

np.set_printoptions(threshold=sys.maxsize)

IMG_SIZE = 32

def create_alphabet(image_path, step, size):
    images = []
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    for i in range(0, h, step):# 180
        for j in range(0, w, step): # 180
            cropped = image[i : i + size, j : j + size] 
            resized_image = cv2.resize(cropped, (32, 32)) 
            name = image_path.split(".")
            cv2.imwrite("./input_" + name[0] + "/{}{}{}.png".format(name[0], i,j), resized_image)

def label_img(img_path):
    match = re.match("(\w+)_\d+\.png", img_path)
    alpha = match.group(1)
    if alpha == "u":
        return np.array([1, 0, 0])
    elif alpha == "e":
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])

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

def main():
    train_data = load_train_data()

    trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    trainLabels = np.array([i[1] for i in train_data])

    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    # model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(BatchNormalization())

    # model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(3, activation = 'softmax'))

    model.summary()
    input()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    model.fit(trainImages, trainLabels, batch_size = 2, epochs = 150, verbose = 1)
    # model.fit(trainImages, trainLabels, batch_size = 2, epochs = 100, verbose = 1)89%

    test_data = load_test_data()    
    testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    testLabels = np.array([i[1] for i in test_data])

    loss, acc = model.evaluate(testImages, testLabels, verbose = 1)
    print(acc * 100)
if __name__ == "__main__":
    main()
