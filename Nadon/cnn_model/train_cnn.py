import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


import tensorflow as tf
from sklearn.model_selection import KFold
from statistics import *
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score

# Image path
DATADIR = '/Users/nadonpanwong/Desktop/CSCI323_Group30/Nadon/testImages/image1.png'
CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# IMG_SIZE = 100
training_data = []

def create_training_data():
    for category in CATEGORIES:
        
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                training_data.append([img_array, class_num])
            except Exception as e:
                pass
            
create_training_data()
print("### " + str(len(training_data)) + " ###")

# IMG_SIZE = 60
# X = []
# y = []

# for features, label in training_data:
#     X.append(features)
#     y.append(label)
    
# X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# X = X.astype('float32') / 255.0

# y = np.array(y)

# model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])

# model.fit(X, y, epochs=5)

# model.save('sudoku_model.h5')