import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.layers import Dense,Conv2D,AveragePooling2D,Flatten


def LeNet5():
    model = models.Sequential()
    
    # c1: Convolution (6 feature map)
    model.add(Conv2D(6,kernel_size=(5,5),activation='tanh',input_shape=(32,32,1),padding='valid'))
    model.add(AveragePooling2D(pool_size=(2,2),strides=2,padding='valid'))

    # c2: Convolution (16 feature map)
    model.add(Conv2D(16,kernel_size=(5,5),padding='valid',activation='tanh'))
    model.add(AveragePooling2D(pool_size=(2,2),strides=2,padding='valid'))

    # Flatten output
    model.add(Flatten())

    #Fully Connected layers
    model.add(Dense(120,activation='tanh'))
    model.add(Dense(84,activation='tanh'))
    model.add(Dense(10,activation='softmax'))

    return model


