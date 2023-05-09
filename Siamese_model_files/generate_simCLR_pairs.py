# partially adapted from https://medium.com/@roushanakrahmat/contrastive-learning-tensorflow-tutorial-c114ff964a48
import numpy as np
from augmentCIFAR import augmentCIFAR
from simCLR_CIFAR import findSims

def generate_pairs_simCLR(x_train, y_train):
    # add augmented versions of images to the dataset
    new_x_train, new_y_train = augmentCIFAR(x_train, y_train)
    
    # build pairs of data
    pairImages = []
    pairLabels = []
    pairPosNeg = []
    
    # load previously generated similarity matrix 
    sims = findSims(x_train, y_train)
    
    # for each image in big training data, find a pos and neg pair 
    for i in range(new_x_train.shape[0]): 
        positives = np.where(sims[i] == 1)
        negatives = np.where(sims[i] == 0)
        
        # add a positive pair to the list by selecting random index
        pos_idx = np.random.choice(positives[0])
        pairImages.append((new_x_train[i], new_x_train[pos_idx]))
        pairLabels.append((new_y_train[i], new_y_train[pos_idx]))
        pairPosNeg.append(1)
        
        # add a negative pair to the list by selecting random index
        neg_idx = np.random.choice(negatives[0])
        pairImages.append((new_x_train[i], new_x_train[neg_idx]))
        pairLabels.append((new_y_train[i], new_y_train[neg_idx]))
        pairPosNeg.append(0)
    
    return np.array(pairImages), np.array(pairLabels), np.array(pairPosNeg)