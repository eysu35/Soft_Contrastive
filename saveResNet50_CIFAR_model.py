from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

####################################################
## Build Model using ResNet50 and Classifier Head ##
####################################################

def build_model():
    # load in ResNet50 model from tensorflow
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256,256,3))

    # add classifier on top of conv_base
    model = models.Sequential()

    # upsample to resize inputs of CIFAR10 from (32x32x3) to (224x224x3)
    model.add(layers.UpSampling2D(size=(2,2)))
    model.add(layers.UpSampling2D(size=(2,2)))
    model.add(layers.UpSampling2D(size=(2,2)))
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

    # train model on CIFAR
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
    return model 

model = build_model()

tf.keras.models.save_model(model, '~/scratch/gpfs/eysu/SoftCL/models/ResNet50+CIFAR_2', overwrite=True, save_format='h5')

model.summary()
