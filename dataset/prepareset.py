#!/usr/bin/env python

################################################################################
#
# Create the testing and training samples
# Usage: pyton prepareset.py size_training files size_testing files first_file_num last_file_num
#
################################################################################

import os, sys
import numpy as np
import h5py

def make_sample_array( training_size, testing_size, folder, bounds ):
    """
    return the list of files for the training and testing samples
    """

    training = []
    testing = []
    sum = 0

    previous_message=""

    f_x = [f for f in os.listdir(folder) if '_x_' in f]
    for file in f_x:

        this_message = 'Complete: %d %%' % int(float(sum)/float(training_size+testing_size)*100.)
        if previous_message != this_message:
            print this_message
            previous_message = this_message

        if sum <= training_size:
            if copy_file( folder+file, "./training" , bounds ):
                sum += 1
        elif sum > training_size and sum < training_size+testing_size:
            if copy_file( folder+file, "./testing" , bounds ):
      	         sum += 1
        else:
            print "All files copied!"
            break

    return

def get_num( file ):
    """
    return the num of a file with strucutre path/to/db_view_*_x_num.npy
    """

    num_and_ext = file.split('_')[-1]
    num =  num_and_ext.split('.')[0]

    return num

def get_path( file ):
    """
    return the path/to/ of a file with strucutre path/to/db_view_*_x_num.npy
    """
    path = ""
    for piece in file.split('/')[:-1]:
        path += piece+"/"

    return path

def get_name( file ):
    """
    return the name of a file with strucutre path/to/db_view_*_num.hdf5
    """

    name = file.split('/')[-1]

    return name

def get_view( file ):

    name = get_name(file)
    view = name.split('_')[2]

    return view

def copy_file( file, folder, bounds ):
    """
    link every file  file into the specified folder
    """

    #check if the files are already on the list
    if int(get_num(file)) <= bounds[0] or int(get_num(file)) > bounds[1]:
        return False
    else:

        statement = "ln -s %s %s" % (file, folder)
        os.system(statement)

        return True


#def checklength( dirname, remove ):
#    """
#   """
#
#    f_x = [f for f in os.listdir(dirname) if '_x_' in f]
#    f_y = [f for f in os.listdir(dirname) if '_y_' in f]
#
#    print " in folder %s: %i _x and %i _y " % ( dirname, len(f_x), len(f_y) )
#
#    if len( f_x ) == len( f_y ):
#        print "all ok!"
#        return
#    elif len(f_x) > len(f_y):
#        longest = f_x
#        shortest = f_y
#        arg = '_y_'
#    else:
#        longest = f_y
#        shortest = f_x
#        arg = '_x_'
#
#    for file in longest:
#        num =  get_num(file)
#        matches = [x for x in shortest if x.endswith(arg+num+'.npy') ]
#        if len(matches) == 0:
#            statement = 'rm '+dirname+file
#            print ' Extra file: '+file
#
#            #remove the files if flag correctly set
#            if remove == 'Yes':
#                print statement
#                os.system(statement)


def remove_links( folder ):
    """
    Remove dangling links
    """

    statement = "rm -f %s" % (folder)

    if not os.path.islink(folder):
        os.system(statement)

    return

def make_array_list(folder):
    '''
    Create a numpy array holding ( file number, array idex ) ntuples to identify
    each patch in the database with an unique address
    '''

    address_list = [] #np.empty(1, dtype=int)

    for file in os.listdir(folder):

        print 'reading '+file

        num = get_num(file)
        try:
            db = h5py.File( folder+file , 'r')
            labels =  db.get('labels')
        except:
            print "can't open file"
            continue

        for key in labels.keys():
            id = int( key.split('-')[-1] )
            address_list.append( (num, id) )

    np.save(folder+"/address_list.npy", address_list)

    return


def main():

    folder="/eos/user/a/ascarpel/CNN/neutrino/hdf5/"
    inputdir=""
    training_size = int(sys.argv[1])
    testing_size = int(sys.argv[2])
    first_file =  int(sys.argv[3])
    last_file = int(sys.argv[4])

    make_sample_array( training_size, testing_size, folder+inputdir, (first_file, last_file) )

    #remove dangling symlinks
    remove_links( "./training/" )
    remove_links( "./testing/"  )

    print "All done"


if __name__ == "__main__":
    main()
