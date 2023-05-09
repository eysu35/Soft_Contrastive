##########################################################
## Use CIFAR10H data to generate soft similarity labels ##
##########################################################

import numpy as np
from augmentCIFAR import augmentCIFAR 

# input: training data (which is CIFAR10 test set)
def findSims(x_train, y_train, softLabels): 
    
    # add augmented versions of images to the dataset
#     new_x_train, new_y_train = augmentCIFAR(x_train, y_train)

    # new matrix to store the cosine similarity between soft labels for each image
    pairwise_sims = np.zeros((softLabels.shape[0], softLabels.shape[0]))

    # find cosine similarity between soft vectors of every pair of test images
    for i in range(softLabels.shape[0]):
        A = softLabels[i]
        for j in range(softLabels.shape[0]):
            B = softLabels[j]
            pairwise_sims[i, j] = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))

    # since we only have the soft label data for the original images and not the augmented version
    # but want to produce a similarity matrix of the same dimensions, tile 4 together 
    sims = np.tile(pairwise_sims, (2, 2))

    return sims