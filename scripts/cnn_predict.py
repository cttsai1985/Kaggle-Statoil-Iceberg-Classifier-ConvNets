#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for predicting imgs using model, which uses 'param.py' to define
structure and to locate weights of model

@author: cttsai (Chia-Ta Tsai), @Oct 2017
"""

import os
#
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt
#
from sklearn.metrics import log_loss
#
import utils
import models
import params
            
###############################################################################
if __name__ == '__main__':
    
    np.random.seed(1017)
    target = 'is_iceberg'
    
    #Load data
    test, test_bands = utils.read_jason(file='test.json', loc='../input/')   
    test_X_dup = utils.rescale(test_bands)
    test_meta = test['inc_angle'].values 
   
    tmp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
    file_weights = '../weights/weights_current.hdf5'
    
    if os.path.isfile(file_weights):
        
        #define and load model
        nb_filters = params.nb_filters
        nb_dense = params.nb_dense
        weights_file  = params.weights_file 
        model = models.get_model(img_shape=(75, 75, 2), f=nb_filters, h=nb_dense)
        model.load_weights(weights_file)
        
        #
        batch_size = params.batch_size_test
        
        if params.validate_before_test:
            
            train, train_bands = utils.read_jason(file='train.json', loc='../input/')   
            train_X = utils.rescale(train_bands)
            train_meta = train['inc_angle'].values
            train_y = train[target].values
            print('\nPredict training data as validation: {} {}'.format(train_X.shape, train_meta.shape), flush=True)
    
            p = model.predict([train_X, train_meta], batch_size=batch_size, verbose=1)
            print('\nValid loss on training data: {}'.format(log_loss(train_y, p)), flush=True)

        print('\nPredict test data: {} {}'.format(test_X_dup.shape, test_meta.shape), flush=True)
        ids = test['id'].values

        #prediction
        pred = model.predict([test_X_dup, test_meta], batch_size=batch_size, verbose=1)
        pred = np.squeeze(pred, axis=-1)
            
        file = 'subm_{}_f{:03d}.csv'.format(tmp, nb_filters)
        print('\nSave to {}'.format(file))
        subm = pd.DataFrame({'id': ids, target: pred})
        subm.to_csv('../submit/{}'.format(file), index=False, float_format='%.6f')
