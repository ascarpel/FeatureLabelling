import argparse
parser = argparse.ArgumentParser(description='Run CNN training on patches with a few different hyperparameter sets.')
parser.add_argument('-c', '--config', help="JSON with script configuration", default='config.json')
parser.add_argument('-o', '--output', help="Output model file name", default='model')
parser.add_argument('-g', '--gpu', help="Which GPU index", default='0')
args = parser.parse_args()

#################### module import and initialization  #########################

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

import h5py

from utils import read_config, get_patch_size, count_events

#######################  save model function  ##################################

def save_model(model, name):
    try:
        with open(name + '_architecture.json', 'w') as f:
            f.write(model.to_json())
        model.save_weights(name + '_weights.h5', overwrite=True)
        return True   # Save successful
    except:
        return False  # Save failed

def sample_len( dir ):
    """
    Return the total number of files composing the sample in dir
    """
    num = 0
    listsubdir = [ subdir for subdir in os.listdir(dir)  ]
    for subdir in listsubdir:
        num += len( os.listdir(dir+subdir) )

    return num

#######################  keras callbacks class  ################################

class RecordHistory(Callback):

    def on_train_begin(self, logs={}):

        #losses
        self.loss = []
        self.em_trk_none_netout_loss = []
        self.michel_netout_loss = []

        #val losses
        self.val_loss = []
        self.em_trk_none_netout_val_loss = []
        self.michel_netout_val_loss = []

        #acc
        self.em_trk_none_netout_acc = []
        self.michel_netout_acc = []

        #val acc
        self.em_trk_none_netout_val_acc = []
        self.michel_netout_val_acc = []

    def on_epoch_end(self, batch, logs={}):

        #loss
        self.loss.append(logs.get('loss'))
        self.em_trk_none_netout_loss.append(logs.get('em_trk_none_netout_loss'))
        self.michel_netout_loss.append(logs.get('michel_netout_loss'))

        #val loss
        self.val_loss.append(logs.get('val_loss'))
        self.em_trk_none_netout_val_loss.append(logs.get('val_em_trk_none_netout_loss'))
        self.michel_netout_val_loss.append(logs.get('val_michel_netout_loss'))

        #acc
        self.em_trk_none_netout_acc.append( logs.get('em_trk_none_netout_acc') )
        self.michel_netout_acc.append( logs.get('michel_netout_acc') )

        #val acc
        self.em_trk_none_netout_val_acc.append( logs.get('val_em_trk_none_netout_acc') )
        self.michel_netout_val_acc.append( logs.get('val_michel_netout_acc') )

    def print_history( self ):
        print self.loss
        print self.em_trk_none_netout_loss
        print self.michel_netout_loss
        print self.val_loss
        print self.em_trk_none_netout_val_loss
        print self.michel_netout_val_loss

        print self.em_trk_none_netout_acc
        print self.michel_netout_acc
        print self.em_trk_none_netout_val_acc
        print self.michel_netout_val_acc

    def save_history( self, outdir ):
        np.save( outdir+'loss.npy' , self.loss )
        np.save( outdir+'em_trk_none_netout_loss.npy' , self.em_trk_none_netout_loss )
        np.save( outdir+'michel_netout_loss.npy' , self.michel_netout_loss )
        np.save( outdir+'val_loss.npy' , self.val_loss )
        np.save( outdir+'em_trk_none_netout_val_loss.npy' , self.em_trk_none_netout_val_loss )
        np.save( outdir+'michel_netout_val_loss.npy' , self.michel_netout_val_loss )

        np.save( outdir+'em_trk_none_netout_acc.npy' , self.em_trk_none_netout_acc)
        np.save( outdir+'michel_netout_acc.npy' , self.michel_netout_acc )
        np.save( outdir+'em_trk_none_netout_val_acc.npy' , self.em_trk_none_netout_val_acc )
        np.save( outdir+'michel_netout_val_acc.npy' , self.michel_netout_val_acc )

>>>>>>> 38519f94dca08b2348b57eaaf318d43f76f44922

#######################  model configuration  ##################################

print 'Reading configuration...'
config = read_config(args.config)

CNN_INPUT_DIR = config['training_on_patches']['input_dir']
# input image dimensions
PATCH_SIZE_W = config['prepare_data_em_track']['patch_size_w']
PATCH_SIZE_D = config['prepare_data_em_track']['patch_size_d']
img_rows, img_cols = PATCH_SIZE_W, PATCH_SIZE_D

batch_size = config['training_on_patches']['batch_size']
nb_classes = config['training_on_patches']['nb_classes']
nb_epoch = config['training_on_patches']['nb_epoch']
n_training = sample_len( "/data/ascarpel/FeatureLabelling/dataset/training/" )
n_testing = sample_len( "/data/ascarpel/FeatureLabelling/dataset/testing/" )

print " Training sample size: %d " % n_training
print " Testing sample size: %d " % n_testing

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
    model.compile(
                  optimizer=sgd,
                  loss={'em_trk_none_netout': 'categorical_crossentropy', 'michel_netout': 'mean_squared_error'},
                  loss_weights={'em_trk_none_netout': 0.1, 'michel_netout': 1.},
                  metrics=['accuracy']
                  )

##########################  callbacks  #########################################

tb = TensorBoard( log_dir=args.output+'/logs',
                  histogram_freq=0,
                  batch_size=batch_size,
                  write_graph=True,
                  write_images=True
                )
history = RecordHistory()

##########################  generator  #########################################

train_gen = ImageDataGenerator(
                rescale=1./255,
                featurewise_center=False, samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0, width_shift_range=0, height_shift_range=0,
                horizontal_flip=True, # randomly flip images
                vertical_flip=False )  # only horizontal flip

test_gen = ImageDataGenerator( rescale=1./255 )

def get_label_3class( num ):
    """
    return a 3 class array with the binary classification of the image
    """
    if num == 3:
        return [1, 0, 0] #track
    elif num == 0 or num==2:
        return [0, 1, 0] #em
    elif num == 1:
        return [0, 0, 1] #none
    else:
        print "get_label_3class: Invalid class option"

def get_label_1class( num ):
    """
    reurn 1 class array with binary classification of the image
    """
    if num == 0:
        return [1] #michel
    elif num == 1 or num == 2 or num == 3 :
        return [0] #nomichel
    else:
        print "get_label_1class: Invalid class option"

def generate_data_generator(generator, folder, b):
    """
    Input generator to the model. Read images from folders, prepare the batch
    and make the output labels in the correct format
    """

    print folder

    gen = generator.flow_from_directory(    directory=folder,
                                            target_size=(img_rows, img_cols),
                                            color_mode="grayscale",
                                            batch_size=b,
                                            class_mode = 'binary',
                                            shuffle=True,
                                            seed=7,
                                            follow_links = True )
    while True:
            #make the dictionary to pass as input to the fit_generator method
            ntuple = gen.next()

            #convert the batch labels array
            em_trk_none = np.asarray([ get_label_3class(num) for num in ntuple[1] ])
            michel_netout = np.asarray([ get_label_1class(num) for num in ntuple[1] ])

            yield {'main_input': ntuple[0]}, {'em_trk_none_netout': em_trk_none, 'michel_netout': michel_netout}

##########################  training  ##########################################

if n_training/batch_size == 0:
    print "training steps not configured! "
elif n_testing/batch_size == 0:
    print "testing steps not configured! "

print 'Fit config:', cfg_name
model.fit_generator(
                     generator=generate_data_generator(train_gen, '/training/training', batch_size  ),
                     validation_data=generate_data_generator(test_gen, './dataset/testing/', batch_size  ),
                     steps_per_epoch=n_training/batch_size,
                     validation_steps=n_testing/batch_size,
                     epochs=nb_epoch,
                     verbose=1,
                     callbacks=[tb, history],
                    )

################################################################################

history.print_history()
history.save_history(args.output)

if save_model(model, args.output + cfg_name):
    print('All done!')
else:
    print('Error: model not saved.')
