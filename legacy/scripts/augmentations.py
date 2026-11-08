#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains some image augmentations by calling 'opencv' and 
'keras.preprocessing.image'. Currently, 4 kinds of augmentations: 
'Flip', 'Rotate', 'Shift', 'Zoom' are available.

@author: cttsai (Chia-Ta Tsai), @Oct 2017
"""

from random import choice
import cv2
import numpy as np
import keras.preprocessing.image as prep

#data augmentations
###############################################################################        
def HorizontalFlip(image, u=0.5, v=1.0):
    
    if v < u:
        image = cv2.flip(image, 1)
    
    return image


def VerticalFlip(image, u=0.5, v=1.0):
    
    if v < u:
        image = cv2.flip(image, 0)
    
    return image


def Rotate90(image, u=0.5, v=1.0):

    if v < u:
        image = np.rot90(image, k=choice([0, 1, 2, 3]), axes=(0, 1))

    return image
    

def Rotate(image, rotate_rg=45, u=0.5, v=1.0):

    if v < u:
        image = prep.random_rotation(image, rg=rotate_rg, 
                                     row_axis=0, col_axis=1, channel_axis=2)

    return image


def Shift(image, width_rg=0.1, height_rg=0.1, u=0.5, v=1.0):

    if v < u:
        image = prep.random_shift(image, wrg=width_rg, hrg=height_rg, 
                                  row_axis=0, col_axis=1, channel_axis=2)

    return image


def Zoom(image, zoom_rg=(0.1, 0.1), u=0.5, v=1.0):

    if v < u:
        image = prep.random_zoom(image, zoom_range=zoom_rg,
                                  row_axis=0, col_axis=1, channel_axis=2)

    return image
