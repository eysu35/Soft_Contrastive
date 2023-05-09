################################
## Adapted from SupCon  paper ##
################################

import numpy as np
from augmentCIFAR import augmentCIFAR 

# input: training data (which is CIFAR10 test set)
def findSims(x_train, y_train): 
    
    # add augmented versions of images to the dataset
    new_x_train, new_y_train = augmentCIFAR(x_train, y_train)
    
    # build a similarity matrix
    sims = np.zeros((new_x_train.shape[0], new_x_train.shape[0]))
    
    # for all entries in the similarity matrix where j is the augmented image corresponding to i
    # or i is the augmented image corresponding to j, set similarity = 1
    for i in range(new_x_train.shape[0]): 
        for j in range(new_x_train.shape[0]): 
            if j == 1:
                sims[i][j] = 1
                
            elif j == i + x_train.shape[0]:
                sims[i][j] = 1
                
            elif j == i - x_train.shape[0]:
                sims[i][j] = 1
                
            # if the images belong to the same class, set similarity = 1
            elif new_y_train[i] == new_y_train[j]:
                sims[i][j] = 1
                
    return sims

