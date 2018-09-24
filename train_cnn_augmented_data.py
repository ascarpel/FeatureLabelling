import argparse
parser = argparse.ArgumentParser(description='Run CNN training on patches with a few different hyperparameter sets.')
parser.add_argument('-c', '--config', help="JSON with script configuration", default='config.json')
parser.add_argument('-o', '--output', help="Output model file name", default='model')
parser.add_argument('-g', '--gpu', help="Which GPU index", default='0')
args = parser.parse_args()

import os
os.environ['KERAS_BACKEND'] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import tensorflow as tf
import keras
if keras.__version__[0] != '2':
    print 'Please use the newest Keras 2.x.x API with the Tensorflow backend'
    quit()
keras.backend.set_image_data_format('channels_last')
keras.backend.set_image_dim_ordering('tf')

import numpy as np
np.random.seed(2017)  # for reproducibility
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
# from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from os.path import exists, isfile, join
import json

from utils import read_config, get_patch_size, count_events, RecordHistory

def save_model(model, name):
    try:
        with open(name + '_architecture.json', 'w') as f:
            f.write(model.to_json())
        model.save_weights(name + '_weights.h5', overwrite=True)
        return True   # Save successful
    except:
        return False  # Save failed

#######################  model configuration  ##################################

print 'Reading configuration...'
config = read_config(args.config)

CNN_INPUT_DIR = config['training_on_patches']['input_dir']
# input image dimensions
PATCH_SIZE_W, PATCH_SIZE_D = get_patch_size(CNN_INPUT_DIR)
img_rows, img_cols = PATCH_SIZE_W, PATCH_SIZE_D

batch_size = config['training_on_patches']['batch_size']
nb_classes = config['training_on_patches']['nb_classes']
nb_epoch = config['training_on_patches']['nb_epoch']

nb_pool = 2 # size of pooling area for max pooling

cfg_name = 'sgd_lorate'

# convolutional layers:
nb_filters1 = 48  # number of convolutional filters in the first layer
nb_conv1 = 10      # 1st convolution kernel size
convactfn1 = 'relu'

maxpool = False   # max pooling between conv. layers

nb_filters2 = 0   # number of convolutional filters in the second layer
nb_conv2 = 7      # convolution kernel size
convactfn2 = 'relu'

drop1 = 0.2

# dense layers:
densesize1 = 128
actfn1 = 'relu'
densesize2 = 32
actfn2 = 'relu'
drop2 = 0.2

#######################  model definition  #####################################

print 'Compiling CNN model...'
with tf.device('/gpu:' + args.gpu):
    main_input = Input(shape=(img_rows, img_cols, 1), name='main_input')

    if convactfn1 == 'leaky':
        x = Conv2D(nb_filters1, (nb_conv1, nb_conv1),
                   padding='valid', data_format='channels_last',
                   activation=LeakyReLU())(main_input)
    else:
        x = Conv2D(nb_filters1, (nb_conv1, nb_conv1),
                   padding='valid', data_format='channels_last',
                   activation=convactfn1)(main_input)

    if nb_filters2 > 0:
        if maxpool:
	    x = MaxPooling2D(pool_size=(nb_pool, nb_pool))(x)
        x = Conv2D(nb_filters2, (nb_conv2, nb_conv2))(x)
        if convactfn2 == 'leaky':
            x = Conv2D(nb_filters2, (nb_conv2, nb_conv2), activation=LeakyReLU())(x)
        else:
            x = Conv2D(nb_filters2, (nb_conv2, nb_conv2), activation=convactfn2)(x)

    x = Dropout(drop1)(x)
    x = Flatten()(x)
    # x = BatchNormalization()(x)

    # dense layers
    x = Dense(densesize1, activation=actfn1)(x)
    x = Dropout(drop2)(x)

    if densesize2 > 0:
        x = Dense(densesize2, activation=actfn2)(x)
        x = Dropout(drop2)(x)

    # outputs
    em_trk_none = Dense(3, activation='softmax', name='em_trk_none_netout')(x)
    michel = Dense(1, activation='sigmoid', name='michel_netout')(x)

    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model = Model(inputs=[main_input], outputs=[em_trk_none, michel])
    model.compile(optimizer=sgd,
                  loss={'em_trk_none_netout': 'categorical_crossentropy', 'michel_netout': 'mean_squared_error'},
                  loss_weights={'em_trk_none_netout': 0.1, 'michel_netout': 1.},
                  metrics=['accuracy'])


##########################  callbacks  #########################################

tb = TensorBoard( log_dir=args.output+'/logs',
                  histogram_freq=0,
                  batch_size=batch_size,
                  write_graph=True,
                  write_images=True
                )
history = RecordHistory()

##########################  generator  #########################################

#datagen = ImageDataGenerator(
#                featurewise_center=False, samplewise_center=False,
#                featurewise_std_normalization=False,
#                samplewise_std_normalization=False,
#                zca_whitening=False,
#                rotation_range=0, width_shift_range=0, height_shift_range=0,
#                horizontal_flip=True, # randomly flip images
#                vertical_flip=False)  # only horizontal flip
#datagen.fit(X_train)

# Implement DataGenerator class inheriting the Sequence object

class DataGenerator( keras.utils.Sequence ):
    """
    Description here
    """

    def __init__( self, list_IDs, batch_size, dim ,path, dirname):
        """ Class initialization """

        self.batch_size = batch_size
        self.list_IDs = list_IDs # holds address ntuples equal for both x and y
        self.list_IDs_temp = [] # holds temps address ntuples equal for both x and y
        self.dim = dim # ntuples with the patch dimension from config file
        self.path = path #initial directory
        self.dirname = dirname    #training or testing

        self.on_epoch_end()

    def __len__( self ):
        """ Denotes the number of batches per epoch (mandatory) """

        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__( self, index ):
        """ Generate one batch of data ( mandatory ) """

        X, Y = self.__data_generation()

        return X, Y

    def on_epoch_end( self ):
        """ Update indeces after each epoch """

        #regenerate the index list
        self.list_IDs_temp = self.list_IDs

    def __get_random( self ):
         """
         Get a random address from the list,
         remove that entry so one will use it only once per epoch
         return filenum and array index
         """

         #get a random number
         index = np.random.randint(0, len(self.list_IDs_temp), 1)

         address = self.list_IDs_temp[index]
         np.delete( self.list_IDs_temp, index )

         num = int(address[0][0])
         id = int(address[0][1])

         return num, id

    def __data_generation( self ):
        """ Generates data containing batch_size samples """

        #Input array
        X = np.zeros( self.dim, dtype=np.float32)

        #Output arrays (NB: dimensions are hardcoded because part of the model )
        EmTrkNone = np.zeros((self.batch_size, 3), dtype=np.int32)
        Michel = np.zeros((self.batch_size, 1), dtype=np.int32)

        for i in range( 0, self.batch_size ):

            num, id = self.__get_random()

            #read all the files associated to it
            fnameX = "db_view_1_x_%d.npy" % num

            fnameY = fnameX.replace('_x_', '_y_')
            X[i] = ( np.load(self.path + '/' + self.dirname + '/' + fnameX, mmap_mode='r') )[id]

            dataY = ( np.load(self.path + '/' + self.dirname + '/' + fnameY, mmap_mode='r') )[id]

            EmTrkNone[i] = [dataY[0], dataY[1], dataY[3]]
            Michel[i] = [dataY[2]]

            #TODO: data augmentation?

        return {'main_input': X}, {'em_trk_none_netout': EmTrkNone, 'michel_netout': Michel}

##########################  training  ##########################################

#training generator
training_address = np.load( CNN_INPUT_DIR + '/' + 'training' + '/' + 'address_list.npy'  )
n_train = len( training_address )

training_generator = DataGenerator( training_address,
                                    batch_size,
                                    ( batch_size, PATCH_SIZE_W, PATCH_SIZE_D ),
                                    CNN_INPUT_DIR,
                                    'training'
                                   )

#testing generator
testing_address = np.load( CNN_INPUT_DIR + '/' + 'testing' + '/' + 'address_list.npy'  )
n_train = len( training_address )

validation_generator = DataGenerator( testing_address,
                                      batch_size,
                                      ( batch_size, PATCH_SIZE_W, PATCH_SIZE_D),
                                      CNN_INPUT_DIR,
                                      'testing'
                                    )

print 'Fit config:', cfg_name
model.fit_generator(
                     generator=training_generator,
                     validation_data=validation_generator,
                     steps_per_epoch=n_train/batch_size, epochs=nb_epoch,
                     verbose=1,
                     callbacks=[tb, history]
                    )

################################################################################

history.print_history()
history.save_history(args.output)

if save_model(model, args.output + cfg_name):
    print('All done!')
else:
    print('Error: model not saved.')
