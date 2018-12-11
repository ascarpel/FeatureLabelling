#!/usr/bin/env python

################################################################################
#
# Create the testing, training and validation samples
# Usage: pyton prepareset.py training_size testing_size validation_size
#
################################################################################
import tensorflow as tf

import os, sys
import numpy as np
import random

class MakeSample():

    def __init__ ( self ):
        #list containig all the files absolute address to use as input
        self.fileslist = []
        self.this_message = ""
        self.previous_message = ""

    def __print_message(self, prog, tot):
        """
        Print progression
        """

        self.this_message = 'Complete: %d %%' % int(float(prog)/float(tot)*100.)
        if self.previous_message != self.this_message:
            print self.this_message
            self.previous_message = self.this_message

    def __get_path(self, file ):
        """
        return the path/to/ of a file with strucutre path/to/db_view_*_x_num.npy
        """
        path = ""
        for piece in file.split('/')[:-1]:
            path += piece+"/"

        return path

    def __get_name(self, file ):
        """
        return the name of a file with strucutre path/to/db_view_*_num.hdf5
        """

        name = file.split('/')[-1]

        return name

    def __get_extension(self, file ):
        """
        Return the name of a file with strucutre path/to/db_view_*_num.hdf5
        """

        extension = file.split('.')[-1]

        return extension

    def __get_labes_from_name( self, name ):
        """
        Get the labels from the file name.
        """

        #extension
        name = name.split('.')[0]

        #get splut array
        spl_buffer = name.split('_')

        #find class names in spl_buffer
        if 'track' in spl_buffer:
            return 'track'
        elif 'shower' in spl_buffer:
            return 'shower'
        elif 'michel' in spl_buffer:
            return 'michel'
        elif 'none' in spl_buffer:
            return 'none'
        else:
            return 0

    def make_list_from_folder( self, dirname ):
        """
        Fill the dirlist with all the .png images stored in the given directory
        """

        #loop over folders
        dirlist = [ dirname+'/'+dir for dir in os.listdir(dirname) if '.tar.gz' not in dir ]

        #put all the files into a single list
        for dir in dirlist:
            print dir
            self.fileslist = self.fileslist + [ dirname+'/'+f for f in os.listdir(dirname) if '.png' in f ]

        #reshuffle the list
        random.shuffle( self.fileslist )

    def __print_num_of_classes( self, list ):
        """
        Print the total number of files processed and how many files per class are present
        """

        tracks = [ file for file in list if '_track_' in file ]
        shower = [ file for file in list if '_shower_' in file ]
        none   = [ file for file in list if '_none_' in file ]
        michel = [ file for file in list if '_michel_' in file ]

        print " All files: %d " % len( list )

        print " Tracks:    %d " % len( tracks )
        print " Showers:   %d " % len( shower )
        print " Michel:    %d " % len( michel )
        print " None:      %d " % len( none )

    def make_list_from_file( self, file ):
        """
        Fill dirlist with all the absolute paths in file
        """

        files = open(file, 'r')
        self.fileslist = [ file.split('\n')[0] for file in files if '.png' in file ]

        #randomize the order
        random.shuffle( self.fileslist )

        #print how many fiels per classes from the list
        self.__print_num_of_classes( self.fileslist )


    def create_links(self, target_dir, sample_size ):
        """
        organize the database inside target_folder
        """

        print 'Create links in folder: %s' % target_dir

        #remove the link previously existing:
        self.remove_links( target_dir, 'all' )

        index = 0
        num = 0
        n_skip = 0
        newlist = []

        count_showers = 0

        if sample_size > len( self.fileslist ):
            sample_size = len( self.fileslist )
            print "resclaed sample size to: %d" % sample_size

        if sample_size==0:
            print "No files for class in folder %s" % target_dir
            return

        while num < sample_size:

            self.__print_message( num, sample_size )

            if '_shower_' in self.fileslist[ num+n_skip ] and count_showers < 0.6*sample_size:
                count_showers += 1
            elif '_shower_' in self.fileslist[ num + n_skip ]:
                n_skip +=1
                continue

            if num+n_skip < len( self.fileslist ):
                fullname = self.fileslist[ num+n_skip ]
            else:
                print "Sample size length reached"
                return

            name = self.__get_name( fullname ) #isolate path
            dirname = self.__get_labes_from_name(name)
            extension = self.__get_extension(name)

            #append num at the end as unique index for each file
            newname = name.split('.')[0]+"_"+str(index)+"."+extension

            if dirname !=0:

                if 'validation/' in target_dir:
                    #copy the file on eos for the validation sample
                    statement = "scp %s %s/%s/%s" % (fullname, target_dir, dirname, newname)
                    os.system(statement)
                    newlist.append( target_dir+"/"+dirname+"/"+newname )
                else:
                    #make a soft link in the given folder
                    statement = "ln -s %s %s/%s/%s" % (fullname, target_dir, dirname, newname)
                    os.system(statement)
                    newlist.append( target_dir+"/"+dirname+"/"+newname )
            else:
                print 'Invalid label in filename!'
                print 'Options are: track, shower, michel, none'

            self.fileslist.remove(fullname)
            index += 1
            num += 1

        print "In folder %s " % target_dir
        self.__print_num_of_classes( newlist )

    def remove_links(self, target_dir, option ):
        """
        Remove dangling links
        """

        listdir = [ dir for dir in os.listdir(target_dir)  ]

        for dir in listdir:

            print 'check links in folder: ' + target_dir+dir

            for file in os.listdir(target_dir+dir):

                statement = "rm -f %s" % (target_dir+dir+'/'+file)

                if option=='invalid':
                    if not os.path.exists(os.readlink(target_dir+dir+'/'+file )):
                        os.system(statement)
                if option=='all':
                        os.system(statement)


class MakeTFRecord(  ):

    def __init__ ( self, dirname, tfrecord_name):

        self.dirname = dirname
        self.record_name = tfrecord_name

    # Helperfunctions to make your feature definition more readable
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _write(self, writer, image, label):

        # Define the features of your tfrecord
        feature = { 'image':  self._bytes_feature(tf.compat.as_bytes(image.tostring())),
                    'label':  self._bytes_feature(tf.compat.as_bytes(label.tostring()))}

        # Serialize to string and write to file
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    def numpy2tfrecord(self):

        # create filewriter
        writer = tf.python_io.TFRecordWriter(self.record_name)

        files = [ f for f in os.listdir(self.dirname) if '_x.npy' in f ]

        for file in files:

            #check if the associated label file exists and import both .npy
            if os.path.exists( self.dirname+file.replace('_x', '_y') ):
                images = np.load(self.dirname+'/'+file)
                labels = np.load( self.dirname+file.replace('_x', '_y') )
            else:
                print "file doesn't exist"

            for i, image in enumerate(images):

                label =  labels[i]

                np.reshape(image, (66, 68, 1))
                np.reshape(label, (4, 1))

                self._write( writer, image, label )

        writer.close()
        sys.stdout.flush()

        return

    def tfrecord2numpy(self):

        #numpy_images = []
        #numpy_labels = []

        with tf.Session() as sess:

            # define your tfrecord again. Remember that you saved your image as a string.
            feature = {'image': tf.FixedLenFeature([], tf.string),
                       'label': tf.FixedLenFeature([], tf.string)}

            # Create a list of filenames and pass it to a queue
            filename_queue = tf.train.string_input_producer([self.record_name], num_epochs=1)

            # Define a reader and read the next record
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            # Decode the record read by the reader
            np.shape(serialized_example)
            np.shape(feature)
            features = tf.parse_single_example(serialized_example, feature)

            # Convert the image data from string back to the numbers
            image = tf.decode_raw(features['image'], tf.float32)

            # Cast label data into int32
            label = tf.decode_raw(features['label'], tf.int32)

            # Reshape image data into the original shape
            image = tf.reshape(image, [66, 68, 1])
            label = tf.reshape(label, [4, 1])

            images, labels = tf.train.shuffle_batch( [image, label],
                                                      batch_size=10,
                                                      capacity=30,
                                                      num_threads=1,
                                                      min_after_dequeue=10
                                                    )

            # Initialize all global and local variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            img, lbl = sess.run([images, labels])
            img = img.astype(np.float32)

            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)
            sess.close()

        return img, lbl


def main():

    #    filelist="/data/ascarpel/images/particlegun/files.list"
    #    training_size = int(sys.argv[1])
    #    testing_size = int(sys.argv[2])
    #    validation_size = int(sys.argv[3])
    #
    #    mysample = MakeSample()
    #    mysample.make_list_from_file( filelist )
    #
    #    #check and remove invalid links
    #    mysample.remove_links( "./training/", "all" )
    #    mysample.remove_links( "./testing/", "all" )
    #    mysample.remove_links( "/eos/user/a/ascarpel/CNN/particlegun/validation/", "all" )
    #
    #    #previously created links are automatically removed
    #    mysample.create_links( "./training/", training_size )
    #    mysample.create_links( "./testing/", testing_size )
    #    mysample.create_links( "/eos/user/a/ascarpel/CNN/particlegun/validation/", validation_size )
    #
    #    #check and remove invalid links
    #    mysample.remove_links( "./training/", "invalid" )
    #    mysample.remove_links( "./testing/", "invalid" )
    #    ##mysample.remove_links( "/eos/user/a/ascarpel/CNN/particlegun/validation/", "invalid" )

    mysample = MakeTFRecord("/data/ascarpel/", "test.tfrecord")
    mysample.numpy2tfrecord()
    images, labels = mysample.tfrecord2numpy()

    print np.shape(images)
    print np.shape(labels)

    for i, image in enumerate(images):

    print "All done"

if __name__ == "__main__":
    main()
