##############################################
## Load, partition, and resize CIFAR10 Data ##
##############################################
def loadData():
    import pickle

    # unpickle the binary files
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    labels = ['airplane',  # index 0
          'automobile',  # index 1
          'bird',  # index 2 
          'cat',  # index 3 
          'deer',  # index 4
          'dog',  # index 5
          'frog',  # index 6 
          'horse',  # index 7 
          'ship',  # index 8 
          'truck']  # index 9
    
    # paths to each batch of data
    batch1 = unpickle("/scratch/gpfs/eysu/src_data/cifar-10-batches-py/data_batch_1")
    batch2 = unpickle("/scratch/gpfs/eysu/src_data/cifar-10-batches-py/data_batch_2")
    batch3 = unpickle("/scratch/gpfs/eysu/src_data/cifar-10-batches-py/data_batch_3")
    batch4 = unpickle("/scratch/gpfs/eysu/src_data/cifar-10-batches-py/data_batch_4")
    batch5 = unpickle("/scratch/gpfs/eysu/src_data/cifar-10-batches-py/data_batch_5")
    meta = unpickle("/scratch/gpfs/eysu/src_data/cifar-10-batches-py/batches.meta")
    test = unpickle("/scratch/gpfs/eysu/src_data/cifar-10-batches-py/test_batch")

    # separate labels and image data from each batch
    y_train1 = batch1[b'labels']
    x_train1 = batch1[b'data']
    y_train2 = batch2[b'labels']
    x_train2 = batch2[b'data']
    y_train3 = batch3[b'labels']
    x_train3 = batch3[b'data']
    y_train4 = batch4[b'labels']
    x_train4 = batch4[b'data']
    y_train5 = batch5[b'labels']
    x_train5 = batch5[b'data']

    # concatenate into big training and testing arrays
    y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5))
    x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4, x_train5), axis=0)
    
    y_test = test[b'labels']
    x_test = test[b'data']
    
    # Further break training data into train / validation sets 
    # put 5000 into validation set and keep remaining 45,000 for train
    (x_train, x_valid) = x_train[1000:], x_train[:1000] 
    (y_train, y_valid) = y_train[1000:], y_train[:1000]

    # reshape data to match dimensions of cifar10.load_data
    x_train = x_train.reshape(49000, 3, 32, 32)
    x_train = x_train.transpose(0, 2, 3, 1)
    x_train = x_train.astype('float32')
    x_train /= 255

    x_valid = x_valid.reshape(1000, 3, 32, 32)
    x_valid = x_valid.transpose(0, 2, 3, 1)
    x_valid = x_valid.astype('float32')
    x_valid /= 255

    x_test = x_test.reshape(10000, 3, 32, 32)
    x_test = x_test.transpose(0, 2, 3, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)
    
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_valid = tf.keras.utils.to_categorical(y_valid, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    
    # preprocess data to convert from RGB -> BGR and to zero center around ImageNet dataset
    x_train = tf.keras.applications.resnet50.preprocess_input(x_train)
    x_valid = tf.keras.applications.resnet50.preprocess_input(x_valid)
    x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
    
    return x_train, x_valid, x_test, y_train, y_valid, y_test, labels