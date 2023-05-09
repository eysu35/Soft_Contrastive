# partially adapted from https://medium.com/@roushanakrahmat/contrastive-learning-tensorflow-tutorial-c114ff964a48
import numpy as np
from softCLSims import findSims

def generate_pairs_softCL(x_train, y_train):
    # add augmented versions of images to the dataset
    new_x_train, new_y_train = augmentCIFAR(x_train, y_train)
    
    # build pairs of data
    pairImages = []
    pairLabels = []
    
    # load previously generated similarity matrix 
    sims = findSims(x_train, y_train)
    
    # for each image in big training data, find a pos and neg pair 
    for i in range(new_x_train.shape[0]): 
        positives = np.where(sims[i] > 0)
        negatives = np.where(sims[i] == 0)
        
        # add a positive pair to the list
        #### PUT THE SIM DATA IN THE PAIRS ??? CAN I USE THESE AS WEIGHTS 
        
        
        pairImages.append((new_x_train[i], new_x_train[np.random.choice(positives)]))
        pairLabels.append(sims[i])
        
        # add a negative pair to the list by selecting random ind
        pairImages.append((new_x_train[i], new_x_train[np.random.choice(negatives)]))
        pairLabels.append(0)
    
    return np.array(pairImages), np.array(pairLabels)
    