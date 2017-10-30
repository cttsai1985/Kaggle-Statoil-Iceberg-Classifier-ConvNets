# A ConvNet soluction for Kaggle Statoil/C-CORE Iceberg Classifier Challenge

This ConvNet solution implemeted for Kaggle Statoil/C-CORE Iceberg Classifier Challenge (https://www.kaggle.com/c/statoil-iceberg-classifier-challenge). It handles multi-inputs which include one meta data input and TWO channels of image data: HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically). Compared to current approaches relying on ImageGenerator which demands one, three or fourth channels of image data, this script could handle images only with two channels and performimg some augmentation. Currently, 'Flip', 'Rotate', 'Shift', 'Zoom' are included and they are perfomed by calling opencv and keras.preprocessing.image. 


From my limited observations, this ConvNet is able to score around 0.19~0.20+ and should have the potential to go deep by further optimizing such as to extend the training epochs, adjust early stopping, tweak optimizer, etc. I would like to thank petrosgk
for his solution implemented for Carvana Image Masking Challenge (https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge). I learnt a lot for how to build a pipeline with augmentation.


---

## Updates

### Update Oct-30, 2017
* Repo opened *

---

## Requirements
* keras 2.0 w/ TF backend
* pandas
* numpy
* sklearn
* cv2
* h5py

---

## Usage

### Data
Place '*train.json*' and '*test.json*' in the '*input*' folder.

### Train, predict and submit at once
Run `python cnn_train.py` in the '*convnets*' folder to training model, make predictions on test data and generate submission.

### Predict and submit
Run `python cnn_predict.py` in the '*convnets*' folder to make predictions on test data and generate submission.
