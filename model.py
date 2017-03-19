import csv
import cv2
import numpy as np
import os
import sklearn

#read csv lines
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

def generator(lines, batch_size=32):
    num_lines = len(lines)
    while 1: #loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            measurements = []
            correction = 0.38 #correction factor for side cameras

            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = 'data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    if(i == 1):
                        measurement = measurement + correction #left camera
                    elif (i == 2):
                        measurement = measurement - correction #right camera
                    measurements.append(measurement)

            augmented_images = []
            augmented_measurements = []

            for image,measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1)) #flip images horizontally
                augmented_measurements.append(measurement*-1.0) #reverse steering measurement

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield shuffle(X_train, y_train)

#compile and train model using the generator function
train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#NVIDIA Network
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) #normalize image
model.add(Cropping2D(cropping=((70,25),(0,0)))) #crop images to remove "noise" data - top and bottom
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_lines)*3, validation_data=validation_generator, nb_val_samples=len(validation_lines), nb_epoch=5, verbose=1)

model.save('model.h5')