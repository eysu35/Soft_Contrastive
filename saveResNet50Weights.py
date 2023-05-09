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

    return conv_base 

model = build_model()

tf.keras.models.save_model(model, '~/scratch/gpfs/eysu/SoftCL/models/ResNet50_weights', overwrite=True, save_format='h5')

model.summary()
