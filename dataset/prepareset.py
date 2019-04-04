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

import progressbar


class MakeSample():

    def __init__ ( self, SEED=None ):

        #list containig all the files absolute address to use as input
        self.fileslist = []
        self.this_message = ""
        self.previous_message = ""

        self.n_tot=0
        self.n_tracks=0
        self.n_showers=0
        self.n_michel=0
        self.n_none=0

        #select a seed
        if SEED != None:
            random.seed(SEED)

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

    def __get_dirnum(self, file ):
        dirname = file.split('/')[-2]
        return dirname.split('_')[-1]

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

    def __checkdir( self, dir ):
        """
        Check if the directory has both data and label
        """

        if '.tar.gz' in dir:
            return False
        if len([ f for f in os.listdir(dir) ]) == 2:
            return True
        else:
            return False

    def __count_classes( self, file ):
        """
        count the classes
        """
        array = np.load(file, 'r+')
        self.n_tot += len(array)
        self.n_tracks += np.sum( [ entry[0] for entry in array  ] )
        self.n_showers += np.sum( [ entry[1] for entry in array  ] )
        self.n_michel += np.sum( [ entry[2] for entry in array  ] )
        self.n_none += np.sum( [ entry[3] for entry in array  ] )

    def __print_classes(self, dir):
        """
        Print the total number of files processed and how many files per class are present
        """
        print " Total Patches in %s: %d " % (dir, self.n_tot)
        print " Tracks:    %d " % self.n_tracks
        print " Showers:   %d " % self.n_showers
        print " Michel:    %d " % self.n_michel
        print " None:      %d " % self.n_none

    def __reset_classes(self):
        """
        Reset to zero
        """
        self.n_tot=0
        self.n_tracks=0
        self.n_showers=0
        self.n_michel=0
        self.n_none=0

    def make_list_from_folder( self, dirname ):
        """
        Fill the dirlist with all the .npy stored in the given directory
        """

        #loop over folders
        dirlist = [ dirname+dir for dir in os.listdir(dirname) if self.__checkdir(dirname+'/'+dir) ]

        #put all the files into a single list
        for dir in dirlist:
            print dir
            self.fileslist = self.fileslist + [ dir+'/'+f for f in os.listdir(dir) if '_y.npy' in f]

        #reshuffle the list
        random.shuffle( self.fileslist )
        np.savetxt('files.list', self.fileslist, fmt='%s')

    def make_list_from_file( self, file ):
        """
        Fill dirlist with all the absolute paths in file
        """
        files = open(file, 'r')
        self.fileslist = [ file.split('\n')[0] for file in files if '_y.npy' in file ]

        #randomize the order
        random.shuffle( self.fileslist )

    def create_links(self, target_dir, sample_size ):
        """
        organize the database inside target_folder
        """

        print 'Create links in folder: %s' % target_dir

        #remove the link previously existing:
        self.remove_links( target_dir, 'all' )

        num = 0
        count_showers = 0

        if sample_size > len( self.fileslist ):
            sample_size = len( self.fileslist )
            print "resclaed sample size to: %d" % sample_size

        #make progression bar
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
        bar = progressbar.ProgressBar(maxval=sample_size, widgets=widgets )
        bar.start()

        while num < sample_size:

            #self.__print_message( num, sample_size )
            bar.update(num)
            self.__count_classes( self.fileslist[num] )

            #make a soft link for the input to the given folder
            fullname = self.fileslist[num].replace('_y.npy', '_x.tfrecord')
            name = self.__get_name(fullname).split('.')[0]
            ext = self.__get_extension(fullname)
            fnum = self.__get_dirnum(fullname)
            newname = name+'_'+fnum+'.'+ext
            statement = "ln -s %s %s" % (fullname, target_dir+newname)
            os.system(statement)

            num += 1

        bar.finish()
        self.__print_classes(target_dir)
        self.__reset_classes()

    def remove_links(self, target_dir, option ):
        """
        Remove dangling links
        """
        print 'check links in folder: ' + target_dir
        for file in os.listdir(target_dir):
            statement = "rm -f %s" % (target_dir+'/'+file)
            if option=='invalid':
                if not os.path.exists(os.readlink(target_dir+'/'+file )):
                    os.system(statement)
            if option=='all':
                os.system(statement)

class MakeTFRecord():

    def __init__ (self, dirname, tfrecord_name):

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

        for f in files:
            #check if the associated label file exists and import both .npy
            if os.path.exists( self.dirname+f.replace('_x', '_y') ):
                images = np.load(self.dirname+f)
                labels = np.load( self.dirname+f.replace('_x', '_y') )
            else:
                print "file doesn't exist"

            for i, image in enumerate(images):

                label =  labels[i]

                np.reshape(image, (66, 68, 1))
                np.reshape(label, (4, 1))

                self._write( writer, image, label)

        writer.close()
        sys.stdout.flush()


    def tfrecord_loop(self, batch_size, sess):

            feature = {'image': tf.FixedLenFeature([], tf.string ),
                       'label': tf.FixedLenFeature([], tf.string)}

            def _parse_record(example_proto):
                """
                Parse .tfrecord files back into image and labels
                """

                example = tf.parse_single_example(example_proto, feature)
                im = tf.decode_raw(example['image'], tf.float32)
                im = tf.reshape(im, (66, 68, 1))

                label = tf.decode_raw(example['label'], tf.int32)
                label = tf.reshape(label, (4, 1))

                return (im, label)

            dataset =  tf.data.TFRecordDataset(self.record_name).map(_parse_record)
            dataset = dataset.batch(batch_size) #no need to suffle

            #make the iterator as one_shot_iterator
            iter = dataset.make_one_shot_iterator()
            el = iter.get_next()

            while True:
                try:
                    ntuple = sess.run(el)
                    yield ntuple
                except tf.errors.OutOfRangeError:
                    print 'Out of range'
                    break

    def tfrecord2numpy(self):

        with tf.Session() as sess:

            # define your tfrecord again. Remember that you saved your image as a string.
            feature = {'image': tf.FixedLenFeature([], tf.string),
                       'label': tf.FixedLenFeature([], tf.string)}

            # Create a list of filenames and pass it to a queue
            filename_queue = tf.train.string_input_producer([self.record_name], num_epochs=1)

            # Define a reader and read the next record
            reader = tf.TFRecordReader()
            _ , serialized_example = reader.read(filename_queue)

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
                                                      batch_size=100,
                                                      capacity=2,
                                                      num_threads=1,
                                                      min_after_dequeue=1
                                                    )

            # Initialize all global and local variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            img, label = sess.run([images, label])
            img = img.astype(np.float32)

            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)
            sess.close()

        return img, label

def main():

    filelist="/eos/user/a/ascarpel/CNN/particlegun/patches/labels.txt"
    training_size = int(sys.argv[1])
    testing_size = int(sys.argv[2])
    validation_size = int(sys.argv[3])

    mysample = MakeSample()
    #mysample.make_list_from_folder( '/eos/user/a/ascarpel/CNN/particlegun/patches/' )
    mysample.make_list_from_file(filelist)

    #previously created links are automatically removed
    mysample.create_links( "./training/", training_size )
    mysample.create_links( "./testing/", testing_size )
    mysample.create_links( "./validation/", validation_size )

    #check and remove invalid links
    mysample.remove_links( "./training/", "invalid" )
    mysample.remove_links( "./testing/", "invalid" )
    mysample.remove_links( "./validation/", "invalid" )

    print "All done"

def checktfrecords():
    mysample = MakeTFRecord("./", "./test.tfrecord")
    with tf.Session() as sess:
        labels = [img for img, label in mysample.tfrecord_loop(batch_size=1, sess=sess)]
        print len(labels)
        print labels

    print "All done"

if __name__ == "__main__":
    main()
    #checktfrecords()
