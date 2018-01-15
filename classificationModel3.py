
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.data_preprocessing import ImagePreprocessing
import tflearn.metrics
import tflearn.metrics


def createModel(parameters):

    img_prep = ImagePreprocessing()
    img_prep.add_samplewise_zero_center()

    imgaug = tflearn.ImageAugmentation()
    imgaug.add_random_flip_leftright()
    imgaug.add_random_flip_updown()
    imgaug.add_random_90degrees_rotation()

    # Building convolutional network
    network = input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep, data_augmentation=imgaug, name='input')
    network = conv_2d(network, parameters.CNN1FeatureSize, parameters.CNN1Size, activation='relu', regularizer="L2")  # padding same
    network=batch_normalization(network)

    if parameters.CNN2Size > 0:
        network = conv_2d(network, parameters.CNN2FeatureSize, parameters.CNN2Size, activation='relu', regularizer="L2")#2. CNN
        network=batch_normalization(network)
    if parameters.CNN3Size > 0:
        network = conv_2d(network, parameters.CNN3FeatureSize, parameters.CNN3Size, activation='relu', regularizer="L2")#3. CNN
        network=batch_normalization(network)
        network = max_pool_2d(network, 2)
    if parameters.CNN4Size > 0:
        network = conv_2d(network, parameters.CNN4FeatureSize, parameters.CNN4Size, activation='relu', regularizer="L2")#4. CNN
        network=batch_normalization(network)
        network = max_pool_2d(network, 2)

        network = conv_2d(network, parameters.CNN4FeatureSize, parameters.CNN4Size, activation='relu', regularizer="L2")#5. CNN
        network=batch_normalization(network)
        network = max_pool_2d(network, 2)
	
    #if parameters.FullyConn1Size > 0:
    #    network = fully_connected(network, parameters.FullyConn1Size, activation='relu',  regularizer="L2")
    #    network=batch_normalization(network)
    #    #network = dropout(network, 0.2)
    if parameters.FullyConn2Size > 0:
        network = fully_connected(network, parameters.FullyConn1Size, activation='relu', regularizer="L2")
        network=batch_normalization(network)
        network = dropout(network, 0.2)
    if parameters.FullyConn2Size > 0:
        network = fully_connected(network, parameters.FullyConn2Size, activation='relu', regularizer="L2")
        network=batch_normalization(network)
        network = dropout(network, 0.2)
 


    g = tflearn.fully_connected(network, 2, activation='softmax')
    g = tflearn.regression(g, optimizer='SGD', loss='categorical_crossentropy', metric='default' , learning_rate=0.01,batch_size=64)
    return g
