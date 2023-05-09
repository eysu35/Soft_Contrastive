#######################################
## Augment the CIFAR10 training data ##
#######################################
# from numpy import expand_dims
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# from matplotlib import pyplot

# duplicate each image in the training set with an augmented version
def augmentCIFAR(x_train, y_train):

    # use an image augmenter to generate new set of training data
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2)
    
    generated = datagen.flow(x_train, batch_size=x_train.shape[0], shuffle=False)
    x_train_aug = np.array(generated.next())

    # concatenate the augmented training images with the original data
    new_x_train = np.concatenate((x_train, x_train_aug))
    new_y_train = np.concatenate((y_train, y_train))
    
    return new_x_train, new_y_train