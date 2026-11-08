#model
nb_filters = 8
nb_dense = 128
#train
batch_size = 64
epochs = 3
#weights
weights_init = '../weights/weights_init.hdf5'
weights_file = '../weights/weights_current.hdf5'
#test
batch_size_test = batch_size
validate_before_test = True
