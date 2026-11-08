#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is describing a ConvNet model, with its parameters could be set 
at 'param.py'. It takes multi-inputs which are TWO-channels and meta information 
such as 'inc_angle'.

@author: cttsai (Chia-Ta Tsai), @Oct 2017
"""
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam


def conv_block(x, nf=8, k=3, s=1, nb=2, p_act='elu'):
    
    for i in range(nb):
        x = Conv2D(filters=nf, kernel_size=(k, k), strides=(s, s),  
                   activation=p_act,
                   padding='same', kernel_initializer='he_uniform')(x)
        
    return x

def dense_block(x, h=32, d=0.5, m=0., p_act='elu'):
    return Dropout(d) (BatchNormalization(momentum=m) (Dense(h, activation=p_act)(x)))


def bn_pooling(x, k=2, s=2, m=0): 
    return MaxPooling2D((k, k), strides=(s, s))(BatchNormalization(momentum=m)(x))
    

def get_model(img_shape=(75, 75, 2), num_classes=1, f=8, h=128):

    """
    This model structure is inspired and modified from the following kernel
    https://www.kaggle.com/knowledgegrappler/a-keras-prototype-0-21174-on-pl
    img_shape: dimension for input image
    f: filters of first conv blocks and generate filters in the following 
       blocks acorrdingly 
    h: units in dense hidden layer
    """ 
    
    #model
    bn_model = 0
    p_activation = 'elu'
    
    #
    input_img = Input(shape=img_shape, name='img_inputs')
    input_img_bn = BatchNormalization(momentum=bn_model)(input_img)
    #
    input_meta = Input(shape=[1], name='angle')
    input_meta_bn = BatchNormalization(momentum=bn_model)(input_meta)
    
    #img_1
    #img_1:block_1
    img_1 = conv_block(input_img_bn, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = bn_pooling(img_1, k=3, s=3, m=0)
    
    #img_1:block_2
    f*=2
    img_1 = Dropout(0.2)(img_1)
    img_1 = conv_block(img_1, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = bn_pooling(img_1, k=3, s=2, m=0)
    
    #img_1:block_3
    f*=2
    img_1 = Dropout(0.2)(img_1)
    img_1 = conv_block(img_1, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = bn_pooling(img_1, k=3, s=3, m=0)
    
    #img_1:block_4
    f*=2
    img_1 = Dropout(0.2)(img_1)
    img_1 = conv_block(img_1, nf=f, k=3, s=1, nb=3, p_act=p_activation)
    img_1 = Dropout(0.2)(img_1)
    img_1 = BatchNormalization(momentum=bn_model)(GlobalMaxPooling2D()(img_1))
    
    #img 2
    img_2 = conv_block(input_img_bn, nf=f, k=3, s=1, nb=6, p_act=p_activation)
    img_2 = Dropout(0.2)(img_2)
    img_2 = BatchNormalization(momentum=bn_model)(GlobalMaxPooling2D()(img_2))
    
    #full connect
    concat = (Concatenate()([img_1, img_2, input_meta_bn]))
    x = dense_block(concat, h=h)
    x = dense_block(x, h=h)
    output = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model([input_img, input_meta],  output)

    model.summary()
    
    return model

if __name__ == '__main__':
    model = get_model()

