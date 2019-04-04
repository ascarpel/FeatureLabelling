################################################################################
#
#
################################################################################
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
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.backend import get_session
from os.path import exists, isfile, join
import json

from utils import read_config
from utils import get_label_1class, get_label_3class
from utils import RecordHistory
from utils import save_model

#######################  read configuration file ###############################

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
n_training=config['training_on_patches']['n_training']
n_testing=config['training_on_patches']['n_testing']

print " Training sample size: %d " % n_training
print " Testing sample size: %d " % n_testing

#######################  model configuration  ##################################

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
                  optimizer='sgd',
                  loss={'em_trk_none_netout': 'categorical_crossentropy', 'michel_netout': 'mean_squared_error'},
                  loss_weights={'em_trk_none_netout': 0.1, 'michel_netout': 1.},
                  metrics=['accuracy']
                  )

#######################  TF Dataset and generator  #############################

def data_generator( filename, batch_size, img_rows, img_cols, sess):

    feature = {'image': tf.FixedLenFeature([], tf.string ),
               'label': tf.FixedLenFeature([], tf.string)}

    def _parse_record(example_proto):
        """
        Parse .tfrecord files back into image and labels
        """

        example = tf.parse_single_example(example_proto, feature)
        im = tf.decode_raw(example['image'], tf.float32)
        im = tf.reshape(im, (img_rows, img_cols, 1))

        label = tf.decode_raw(example['label'], tf.int32)
        label = tf.reshape(label, (4, 1))

        return (im, label)

    dataset =  tf.data.TFRecordDataset(filename).map( _parse_record )
    dataset = dataset.shuffle(1000).batch(batch_size).repeat()
    #add data augmentation

    #make the iterator as one_shot_iterator
    iter = dataset.make_one_shot_iterator()
    el = iter.get_next()

    while True:
        try:
            ntuple = sess.run(el)
            em_trk_none = np.asarray([ get_label_3class(num) for num in ntuple[1] ])
            michel_netout = np.asarray([ get_label_1class(num) for num in ntuple[1] ])
            yield {'main_input': ntuple[0]}, {'em_trk_none_netout': em_trk_none, 'michel_netout': michel_netout}
        except tf.errors.OutOfRangeError:
            print 'Out of range'
            break

######################### Callbacks ############################################

history = RecordHistory()
tb = TensorBoard( log_dir=args.output+'/logs',
                  histogram_freq=0,
                  batch_size=batch_size,
                  write_graph=True,
                  write_images=True
                )

if n_training/batch_size == 0:
    print "training steps not configured! "
elif n_testing/batch_size == 0:
    print "testing steps not configured! "

######################### Model fit ############################################

training_input=[CNN_INPUT_DIR+'training/'+file.strip() for file in os.listdir(CNN_INPUT_DIR+'training/') if '.tfrecord' in file]
testing_input=[CNN_INPUT_DIR+'testing/'+file.strip() for file in os.listdir(CNN_INPUT_DIR+'testing/') if '.tfrecord' in file]

with tf.Session() as sess:

    print 'Fit config:', cfg_name
    model.fit_generator(  generator=data_generator(training_input, batch_size, PATCH_SIZE_W, PATCH_SIZE_D, sess ),
                          validation_data=data_generator(testing_input, batch_size, PATCH_SIZE_W, PATCH_SIZE_D, sess ),
                          steps_per_epoch=n_training/batch_size,
                          validation_steps=n_testing/batch_size,
                          epochs=nb_epoch,
                          verbose=1,
                          workers=0,
                          callbacks=[tb, history],
                         )

######################### save history #########################################

    history.print_history()
    history.save_history(args.output)

    if save_model(model, args.output + cfg_name):
        print('All done!')
    else:
        print('Error: model not saved.')

print "All done"
