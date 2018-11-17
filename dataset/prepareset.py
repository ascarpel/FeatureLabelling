#!/usr/bin/env python

################################################################################
#
# Create the testing, training and validation samples
# Usage: pyton prepareset.py training_size testing_size validation_size
#
################################################################################

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

            fullname = self.fileslist[num]

            if '_shower_' in fullname and count_showers > 0.4*sample_size:
                n_skip =+1
                if num+n_skip < len( self.fileslist ):
                    fullname = self.fileslist[num + n_skip]
                else:
                    print "Sample size length reached"
                    return
            elif '_shower_' in fullname:
                count_showers += 1
            else:
                n_skip = 0

            name = self.__get_name( fullname ) #isolate path
            dirname = self.__get_labes_from_name(name)
            extension = self.__get_extension(name)

            #append num at the end as unique index for each file
            newname = name.split('.')[0]+"_"+str(index)+"."+extension

            if dirname !=0:
                #make a soft link in the given folder
                statement = "ln -s %s %s/%s/%s" % (fullname, target_dir, dirname, newname)
                os.system(statement)
                newlist.append( target_dir+"/"+dirname+"/"+newname )
            else:
                print 'Invalid label in filename!'
                print 'Options are: track, shower, michel, none'

            print "In folder %s " % target_dir
            self.__print_num_of_classes( newlist )

            self.fileslist.remove(fullname)
            index += 1
            num += 1

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

def main():

    filelist="/eos/user/a/ascarpel/CNN/neutrino/images/files.list"
    training_size = int(sys.argv[1])
    testing_size = int(sys.argv[2])
    validation_size = int(sys.argv[3])

    mysample = MakeSample()
    mysample.make_list_from_file( filelist )

    #check and remove invalid links
    mysample.remove_links( "./training/", "all" )
    mysample.remove_links( "./testing/", "all" )
    mysample.remove_links( "/eos/user/a/ascarpel/CNN/neutrino/validation/", "all" )

    #previously created links are automatically removed
    mysample.create_links( "./training/", training_size )
    mysample.create_links( "./testing/", testing_size )
    mysample.create_links( "/eos/user/a/ascarpel/CNN/neutrino/validation/", validation_size )

    #check and remove invalid links
    mysample.remove_links( "./training/", "invalid" )
    mysample.remove_links( "./testing/", "invalid" )
    mysample.remove_links( "/eos/user/a/ascarpel/CNN/neutrino/validation/", "invalid" )

    print "All done"

if __name__ == "__main__":
    main()
