#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for training a ConvNet model, which its structure is defined 
at 'models.py' with some parameters of structure and weights' location is 
set at 'param.py'. 

The ConvNet model takes multi-inputs: 1) TWO-channel with capability to perform 
augmentations from 'augmentations.py' and 2) meta info such as 'inc_angle'. 
Four types of augmentations: 'Flip', 'Rotate', 'Shift', 'Zoom' are available.

@author: cttsai (Chia-Ta Tsai), @Oct 2017
"""
import os
#
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt
#
from random import shuffle, uniform, seed
#evaluation
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
#
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#
import augmentations as aug
import utils
import models
import params

###############################################################################

def data_generator(data=None, meta_data=None, labels=None, batch_size=16, augment={}, opt_shuffle=True):
    
    indices = [i for i in range(len(labels))]
    
    while True:
        
        if opt_shuffle:
            shuffle(indices)
        
        x_data = np.copy(data)
        x_meta_data = np.copy(meta_data)
        x_labels = np.copy(labels)
        
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            sel_indices = indices[start:end]
            
            #select data
            data_batch = x_data[sel_indices]
            xm_batch = x_meta_data[sel_indices]
            y_batch = x_labels[sel_indices]
            x_batch = []
            
            for x in data_batch:
                 
                #augment                               
                if augment.get('Rotate', False):
                    x = aug.Rotate(x, u=0.1, v=np.random.random())
                    x = aug.Rotate90(x, u=0.1, v=np.random.random())

                if augment.get('Shift', False):
                    x = aug.Shift(x, u=0.05, v=np.random.random())

                if augment.get('Zoom', False):
                    x = aug.Zoom(x, u=0.05, v=np.random.random())

                if augment.get('Flip', False):
                    x = aug.HorizontalFlip(x, u=0.5, v=np.random.random())
                    x = aug.VerticalFlip(x, u=0.5, v=np.random.random())

                x_batch.append(x)
                
            x_batch = np.array(x_batch, np.float32)
            
            yield [x_batch, xm_batch], y_batch
            

###############################################################################
if __name__ == '__main__':
    
    np.random.seed(1017)
    target = 'is_iceberg'
    
    #Load data
    train, train_bands = utils.read_jason(file='train.json', loc='../input/')
    test, test_bands = utils.read_jason(file='test.json', loc='../input/')
    
    #target
    train_y = train[target].values
    split_indices = train_y.copy()
    
    #data set
    train_X = utils.rescale(train_bands)
    train_meta = train['inc_angle'].values
    test_X_dup = utils.rescale(test_bands)
    test_meta = test['inc_angle'].values

    #training keras
    #model
    nb_filters = params.nb_filters
    nb_dense = params.nb_dense
    weights_file = params.weights_file
    model = models.get_model(img_shape=(75, 75, 2), f=nb_filters, h=nb_dense)
    weights_init = params.weights_init
    model.save(weights_init)
    #training
    epochs = params.epochs
    batch_size = params.batch_size
    print('epochs={}, batch={}'.format(epochs, batch_size), flush=True)
    opt_augments = {'Flip': False, 'Rotate': False, 'Shift': False, 'Zoom': False}
    opt_augments['Flip'] = True
    opt_augments['Rotate'] = True
    opt_augments['Shift'] = True
    opt_augments['Zoom'] = True    
    print(opt_augments)

    #train, validataion split
    test_ratio = 0.159
    nr_runs = 1
    split_seed = 25
    kf = StratifiedShuffleSplit(n_splits=nr_runs, test_size=test_ratio, train_size=None, random_state=split_seed)

    #training, evaluation, test and make submission
    for r, (train_index, valid_index) in enumerate(kf.split(train, split_indices)):

        tmp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
        
        y1, y2 = train_y[train_index], train_y[valid_index]
        x1, x2 = train_X[train_index], train_X[valid_index]
        xm1, xm2 = train_meta[train_index], train_meta[valid_index]

        print('splitted: {0}, {1}'.format(x1.shape, x2.shape), flush=True)
        print('splitted: {0}, {1}'.format(y1.shape, y2.shape), flush=True)
        ################################
        if r > 0:
            model.load_weights(weights_init)
        
        #optim = SGD(lr=0.005, momentum=0.0, decay=0.002, nesterov=True)
        optim = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002)
        
        model.compile(optimizer=optim, loss="binary_crossentropy", metrics=["accuracy"])
        #call backs
        earlystop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, min_delta=1e-4, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, verbose=1, epsilon=1e-4, mode='min')
        model_chk = ModelCheckpoint(monitor='val_loss', filepath=weights_file, save_best_only=True, save_weights_only=True, mode='min')
        
        callbacks = [earlystop, reduce_lr_loss, model_chk, TensorBoard(log_dir='../logs')]
        ##########
       
        model.fit_generator(generator=data_generator(x1, xm1, y1, batch_size=batch_size, augment=opt_augments),
                            steps_per_epoch= np.ceil(8.0 * float(len(y1)) / float(batch_size)),
                            epochs=epochs,
                            verbose=2,
                            callbacks=callbacks,
                            validation_data=data_generator(x2, xm2, y2, batch_size=batch_size),
                            validation_steps=np.ceil(8.0 * float(len(y2)) / float(batch_size)))


        if os.path.isfile(weights_file):

            model.load_weights(weights_file)
            
            p = model.predict([x2, xm2], batch_size=batch_size, verbose=1)
            print('\n\nEvaluate loss in validation data: {}'.format(log_loss(y2, p)), flush=True)

            p = model.predict([x1, xm1], batch_size=batch_size, verbose=1)
            print('\n\nEvaluate loss in training data: {}'.format(log_loss(y1, p)), flush=True)
            
            print('\nPredict...', flush=True)
            ids = test['id'].values

            #prediction
            pred = model.predict([test_X_dup, test_meta], batch_size=batch_size, verbose=1)
            pred = np.squeeze(pred, axis=-1)
            
            file = 'subm_{}_f{:03d}.csv'.format(tmp, nb_filters)
            subm = pd.DataFrame({'id': ids, target: pred})
            subm.to_csv('../submit/{}'.format(file), index=False, float_format='%.6f')

