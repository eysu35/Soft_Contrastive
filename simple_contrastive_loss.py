# Adapted from https://medium.com/@roushanakrahmat/contrastive-learning-tensorflow-tutorial-c114ff964a48
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

def ids2Sims(ids, sims, bsz): 
    batchSims = np.zeros((bsz, bsz))
    for i in range(bsz):
        batchSims[i] = sims[i][ids]
    return batchSims

def embeddings2dists(embeddings, bsz):
    pairwise_dists = np.zeros((bsz, bsz))

    # find cosine similarity between the embedding vectors for every image in the batch
    for i in range(bsz):
        A = embeddings[i]
        for j in range(bsz):
            B = embeddings[j]
            pairwise_dists[i, j] = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
            
    return pairwise_dists


def contrastive_loss(params):
    sims, embed_layer, bsz = params
    def loss(y_true, y_pred):
        ''' 
        y_true: true labels of the images
        y_pred: predicted labels of the images
        sims: similarity matrix that we are using
        embeddings: hidden layer, type 'Tensor' with dimensions (bsz, dim)
        bsz = batch size'''

        # calculate the categorical cross_entropy loss between the true and predicted labels
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                 reduction=tf.keras.losses.Reduction.SUM)
        bce_loss = bce(y_true, y_pred)

        # calculate the contrastive loss
        ids = tf.argmax(y_true,1)

        # get the similarity scores of the images in the batch
        if len(ids) != bsz:
            batchSims = ids2Sims(ids, sims, len(ids))
        else:
            batchSims = ids2Sims(ids, sims, bsz)

        # find the distances between embeddings for these image ids
        embeddings, _ = embed_layer.get_weights() #128x64
        pairwise_dists = embeddings2dists(embeddings, bsz)

        # weight the distances by their similarity scores
        # if the batch does not contain enough images, ignore the contrastive loss
        # since we run into dimension errors
        if len(ids) != bsz:
            contrastive_loss = 0
            
        else: 
            con_loss = np.multiply(batchSims, pairwise_dists)
            con_loss_norm = (con_loss-np.min(con_loss))/(np.max(con_loss)-np.min(con_loss))
            contrastive_loss = np.sum(con_loss_norm)
            
            # reduce the contribution from the contrastive loss so that it
            # does not dominate the overall loss
            contrastive_loss = 0.01 * contrastive_loss 

        #######                                                              #######
        # total loss is the sum of the cross entropy loss and the contrastive loss##
        #######                                                              #######
        loss = tf.reduce_mean(bce_loss + contrastive_loss)

        return loss

    return loss
