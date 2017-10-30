#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts contains some data processing functions

@author: (Chia-Ta Tsai), @Oct 2017
"""

import pandas as pd
import numpy as np


def rescale(imgs): return imgs / 100. +  0.5

def read_jason(file='', loc='../input'):

    df = pd.read_json('{}/{}'.format(loc, file))
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    #print(df['inc_angle'].value_counts())
    
    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)

    bands = np.stack((band1, band2), axis=-1)
    del band1, band2
    
    return df, bands

