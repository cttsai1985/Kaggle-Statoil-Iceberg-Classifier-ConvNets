# A ConvNet for Kaggle Statoil/C-CORE Iceberg Classifier Challenge

This ConvNet solution is written for Kaggle Statoil/C-CORE Iceberg Classifier Challenge (https://www.kaggle.com/c/statoil-iceberg-classifier-challenge). It handles multi-inputs which are one meta data input and one image input for image data with TWO channels: HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically). Compared to recent approaches relying on ImageGenerator which demands channels of image data restricted to one, three or fourth, this script could performing some augmentations on images only consisted of two channels or other numbers of channels. Currently, four types of augmentations such as 'Flip', 'Rotate', 'Shift', 'Zoom' are provided and they are perfomed by calling 'opencv' and 'keras.preprocessing.image'. 


From my limited observations, this ConvNet is able to score around 0.194~0.205+ and should have the potential to go deep by further optimizing such as to extend the training epochs, adjust early stopping, tweak optimizer, etc. Last but not least, I would like to thank petrosgk for his solution implemented for Carvana Image Masking Challenge (https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge). I learnt a lot for how to build a pipeline with augmentation from there.


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
Run `python cnn_train.py` in the '*scripts*' folder to training model, make predictions on test data and generate submission.

### Predict and submit
Run `python cnn_predict.py` in the '*scripts*' folder to make predictions on test data and generate submission.
