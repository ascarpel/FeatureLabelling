#!/usr/bin/env python

################################################################################
#
# Create the testing and training samples
# Usage: pyton prepareset.py
#
################################################################################
import os, sys
import numpy as np
import random

def untar_all( folder ):
    """
    Untar all the files in folder
    """

    # Write here how to untar the files
    # Modify the names of the untar files so they will feature the tar filename
    # Erase tar folder

def make_files_list( folder ):
    """
    Make a .txt file with the absolute path of the patches
    required for the training, testing and validation
    """

    # Read the patches names form their location
    # Apply some sorting criteria of selection ( eg. shuffle, or select a limited
    # number of file for each class
    # Save selected patches absolute paths in .txt file

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
        Fill the list with all the .png images stored in the given directory
        """

        self.fileslist = [ dirname+'/'+f for f in os.listdir(dirname) if '.png' in f ]
        random.shuffle( self.fileslist )

    def make_list_from_file( self, file ):
        """
        Fill the list starting from a .txt file
        """

    def create_links(self, target_dir, sample_size ):
        """
        organize the database inside target_folder
        """

        print 'Create links in folder: %s' % target_dir

        #remove the link previously existing:
        self.remove_links( target_dir, 'all' )

        num = sample_size

        if num > len( self.fileslist ):
            num = len( self.fileslist )
            print "resclaed input to: %d" % num

        while num >= 0:

            self.__print_message( (sample_size-num), sample_size )

            fullname = self.fileslist[num]
            name = self.__get_name(fullname) #isolate path
            dirname = self.__get_labes_from_name(name)

            if dirname !=0:
                #make a soft link in the given folder
                statement = "ln -s %s %s" % (fullname, target_dir+'/'+dirname)
                os.system(statement)
            else:
                print 'Invalid label in filename!'
                print 'Options are: track, shower, michel, none'

            self.fileslist.remove(fullname)
            num -= 1

    def remove_links(self, target_dir, option ):
        """
        Remove dangling links
        """

        listdir = [ dir for dir in os.listdir(target_dir)  ]

        for dir in listdir:

            print 'check for links in folder: ' + target_dir+dir

            for file in os.listdir(target_dir+dir):

                statement = "rm -f %s" % (target_dir+dir+'/'+file)

                if option=='invalid':
                    if not os.path.islink( target_dir+dir+'/'+file ):
                        os.system(statement)
                if option=='all':
                        os.system(statement)

def main():

    folder="/eos/user/a/ascarpel/CNN/neutrino/images/"
    training_size = int(sys.argv[1])
    testing_size = int(sys.argv[2])

    mysample = MakeSample( )
    mysample.make_list_from_folder( folder )

    mysample.create_links( "./training/", training_size )
    mysample.create_links( "./testing/", testing_size )

    #check and remove invalid links
    mysample.remove_links( "./training/", "invalid" )
    mysample.remove_links( "./testing/", "invalid" )

    print "All done"


if __name__ == "__main__":
    main()
