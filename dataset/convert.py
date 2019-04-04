################################################################################
# Covert input numpy array to tfrecord. Erase previous numpy file
# Usage: pyton convert.py targetdir
################################################################################

import os
import re
from prepareset import MakeTFRecord

import numpy as np

def printanal(dictionary):
    """Print analytics"""
    for key, value in dictionary.iteritems():
        print key, value

def doanal(array, dictionary):
    """Get analyticis stored in the dictionaries"""
    dictionary['track']+=np.sum( [ entry[0] for entry in array  ] )
    dictionary['shower']+= np.sum( [ entry[1] for entry in array  ] )
    dictionary['michel']+= np.sum( [ entry[2] for entry in array  ] )
    dictionary['none']+= np.sum( [ entry[3] for entry in array  ] )

def create_file(filename, option):
    if os.path.isfile(filename):
        os.remove(filename)
    record=open(filename, option)
    return record

def main():

    directory = '/eos/user/a/ascarpel/CNN/particlegun/patches/'
    goodfiles=open('/eos/user/a/ascarpel/CNN/particlegun/patches/goodpatches.txt', 'r')
    gooddirs= [subdir.strip() for subdir in goodfiles]

    tfrecods = create_file('/eos/user/a/ascarpel/CNN/particlegun/patches/tfrecords.txt', 'w')
    labelsfile = create_file('/eos/user/a/ascarpel/CNN/particlegun/patches/labels.txt', 'w')

    #dictionary where to store analyticis
    pionsdb = {'track':0, 'shower':0, 'michel':0, 'none':0}
    protonsdb = {'track':0, 'shower':0, 'michel':0, 'none':0}
    particledb = {'pions':pionsdb, 'protons':protonsdb}

    for subdir in gooddirs[0:100]:
        files = [file.strip() for file in os.listdir(subdir)]

        print subdir

        #write tfrecord to file dont reprocess the folder if tfrecrod exists
        tfrecordfile = [x for x in files if re.search('.tfrecord', x)]
        if len(tfrecordfile) > 0:
            tfrecods.write(subdir+'/'+tfrecordfile[0]+'\n')
        else:
            input = [x for x in files if re.search('_x.npy', x)]
            if len(input) > 0:
                recordname= subdir+'/'+(input[0].replace('.npy', '.tfrecord'))
                mysample = MakeTFRecord(subdir+'/', recordname)
                mysample.numpy2tfrecord()
                tfrecods.write(recordname+'\n')

        #find in folder _y.npy
        labels = [x for x in files if re.search('_y.npy', x)]
        if len(labels) > 0:
           particle = labels[0].split('_')[1]
           doanal(np.load(subdir+'/'+labels[0]) ,particledb[particle])
           labelsfile.write(subdir+'/'+labels[0]+'\n')

    for key, value in particledb.iteritems():
        print key
        printanal(value)

    goodfiles.close()
    tfrecods.close()
    labelsfile.close()

    print "All done"

if __name__ == "__main__":
    main()
