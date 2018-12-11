import argparse
parser = argparse.ArgumentParser(description='Run CNN training on patches with a few different hyperparameter sets.')
parser.add_argument('-c', '--config', help="JSON with script configuration", default='config.json')
parser.add_argument('-m', '--model', help="input CNN model name (saved in JSON and h5 files)", default='cnn_model')
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

############################  configuration  ###################################

print 'Reading configuration...'

config = read_config(args.config)

cfg_name = args.model

PATCH_SIZE_W = config['prepare_data_em_track']['patch_size_w']
PATCH_SIZE_D = config['prepare_data_em_track']['patch_size_d']
img_rows, img_cols = PATCH_SIZE_W, PATCH_SIZE_D
input_dir = config['training_on_patches']['validation_dir']

##############################  Read data  #####################################
from PIL import Image

def load_images( filename ):
    return np.array( Image.open( filename ) )

classes = [ 'track', 'shower', 'michel', 'none' ]
imagedb = {}

for c in classes:
    print c
    images = [ load_images( input_dir+c+'/'+file ) for file in os.listdir( input_dir+c+'/' ) ]
    imagedb[c] = images #save in dictionary

print imagedb['shower'][1]
################################  Load model  #################################
#print 'Import CNN model...'
#with tf.device('/gpu:' + args.gpu):
#    model = load_model(cfg_name)
