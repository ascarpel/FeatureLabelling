import argparse
parser = argparse.ArgumentParser(description='Run CNN training on patches with a few different hyperparameter sets.')
parser.add_argument('-c', '--config', help="JSON with script configuration", default='config.json')
parser.add_argument('-m', '--model', help="input CNN model name (saved in JSON and h5 files)", default='cnn_model')
parser.add_argument('-o', '--output', help="output CNN model name (saved in JSON and h5 files)", default='cnn_model_out')
parser.add_argument('-g', '--gpu', help="Which GPU index", default='0')
args = parser.parse_args()

import os
os.environ['KERAS_BACKEND'] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import tensorflow as tf
import keras
if keras.__version__[0] != '2':
    print 'Please use the newest Keras 2.x.x API with the Tensorflow backend'
    quit()
keras.backend.set_image_data_format('channels_last')
keras.backend.set_image_dim_ordering('tf')

import numpy as np
np.random.seed(2017)  # for reproducibility
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from os.path import exists, isfile, join
import json

from utils import read_config, get_patch_size, count_events, shuffle_in_place, RecordHistory

def load_model(name):
    with open(name + '_architecture.json') as f:
        model = model_from_json(f.read())
    model.load_weights(name + '_weights.h5')
    return model

def save_model(model, name):
    try:
        with open(name + '_architecture.json', 'w') as f:
            f.write(model.to_json())
        model.save_weights(name + '_weights.h5', overwrite=True)
        return True   # Save successful
    except:
        return False  # Save failed

#######################  configuration  #############################
print 'Reading configuration...'

config = read_config(args.config)

cfg_name = args.model
out_name = args.output

CNN_INPUT_DIR = config['training_on_patches']['input_dir']
# input image dimensions
PATCH_SIZE_W, PATCH_SIZE_D = get_patch_size(CNN_INPUT_DIR)
img_rows, img_cols = PATCH_SIZE_W, PATCH_SIZE_D

batch_size = config['training_on_patches']['batch_size']
nb_epoch = config['training_on_patches']['nb_epoch']
nb_classes = config['training_on_patches']['nb_classes']

######################  CNN commpilation  ###########################
print 'Compiling CNN model...'
with tf.device('/gpu:' + args.gpu):
    model = load_model(cfg_name)

    sgd = SGD(lr=0.002, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(
                  optimizer=sgd,
                  loss={'em_trk_none_netout': 'categorical_crossentropy', 'michel_netout': 'mean_squared_error'},
                  loss_weights={'em_trk_none_netout': 0.1, 'michel_netout': 1.0},
                  metrics=['accuracy']
                 )
##########################  callbacks  #########################################

tb = TensorBoard( log_dir=args.output+'/logs',
                  histogram_freq=X_train.shape[0],
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

         view = int(address[0][0])
         num = int(address[0][1])
         id = int(address[0][2])

         return view, num, id

    def __data_generation( self ):
        """ Generates data containing batch_size samples """

        #Input array
        X = np.zeros( ( self.batch_size, self.dim[0], self.dim[1], 1) , dtype=np.float32)

        #Output arrays (NB: dimensions are hardcoded because part of the model )
        EmTrkNone = np.zeros((self.batch_size, 3), dtype=np.int32)
        Michel = np.zeros((self.batch_size, 1), dtype=np.int32)

        for i in range( 0, self.batch_size ):

            #get random numbers and read all the files associated to it
            view, num, id = self.__get_random()

            fname = "db_view_%d_%d.hdf5" % (view, num)

            db = h5py.File( self.path+'/'+self.dirname+'/'+fname , 'r')
            input_dataset_name = 'data/data_%d' % id
            label_dataset_name = 'labels/label_%d' % id

            #inport input data
            dataX = db.get( input_dataset_name )
            X[i] = np.asarray( dataX ).reshape( self.dim[0], self.dim[1], 1 )

            #inport output label
            dataY = db.get( label_dataset_name )
            EmTrkNone[i] = [dataY[0], dataY[1], dataY[3]]
            Michel[i] = [dataY[2]]

            db.close()

            #TODO: data augmentation?

        return {'main_input': X}, {'em_trk_none_netout': EmTrkNone, 'michel_netout': Michel}

##########################  training  ##########################################

#training generator
training_address = np.load( CNN_INPUT_DIR+'/training/address_list.npy'  )
n_train = len( training_address )

training_generator = DataGenerator( training_address,
                                    batch_size,
                                    ( PATCH_SIZE_W, PATCH_SIZE_D ),
                                    CNN_INPUT_DIR,
                                    'training'
                                   )

#testing generator
testing_address = np.load( CNN_INPUT_DIR+'/testing/address_list.npy' )
n_train = len( training_address )

validation_generator = DataGenerator( testing_address,
                                      batch_size,
                                      ( PATCH_SIZE_W, PATCH_SIZE_D ),
                                      CNN_INPUT_DIR,
                                      'testing'
                                    )

print 'Fit config:', cfg_name
model.fit_generator(
                     generator=training_generator,
                     validation_data=validation_generator,
                     steps_per_epoch=n_train/batch_size, epochs=nb_epoch,
                     verbose=1,
                     callbacks=[tb, history],
                     use_multiprocessing=True,
                     workers=2
                    )

################################################################################


history.print_history()
history.save_history(args.output)

if save_model(model, args.output + cfg_name.split('/')[-1]):
    print('All done!')
else:
    print('Error: model not saved.')
