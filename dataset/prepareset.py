#!/usr/bin/env python

################################################################################
#
# Create the testing and training samples
# Usage: pyton prepareset.py size_training files size_testing files first_file_num last_file_num
#
################################################################################

import os, sys
import numpy as np

def make_file_list( folder):
    """
    Dump a list of files used already for the network training.
    One can cancel files already in use
    """

    #fetch the files in all the folders and concatenate the two arrays
    f = [f for f in os.listdir(folder) ]

    return f


def make_sample_array( training_size, testing_size, folder, bounds, list):
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
            if copy_file( folder+file, "./training" , bounds, list):
                sum += 1
        elif sum > training_size and sum < training_size+testing_size:
            if copy_file( folder+file, "./testing" , bounds, list ):
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
    return the name of a file with strucutre path/to/db_view_*_x_num.npy
    """

    name = file.split('/')[-1]

    return name

def get_view( file ):

    name = get_name(file)
    view = name.split('_')[2]

    return view

def copy_file( file, folder, bounds, list ):
    """
    copy every '_x' file and the respective '_y' file into the specified folder
    """

    #get the associated y array
    yfile = "%sdb_view_%s_y_%s.npy" % ( get_path(file), get_view(file), get_num(file) )

    #check if the files are already on the list
    if int(get_num(file)) <= bounds[0] or int(get_num(file)) > bounds[1]:
        return False

    if get_name(file) in list:

        statement = "ln -s %s %s" % (file, folder)
        os.system(statement)

        statement = "ln -s %s %s" % (yfile, folder)
        os.system(statement)

        return True
    else:
        return False

def checklength( dirname, remove ):
    """
    Check if every input array has the associated output
    """

    f_x = [f for f in os.listdir(dirname) if '_x_' in f]
    f_y = [f for f in os.listdir(dirname) if '_y_' in f]

    print " in folder %s: %i _x and %i _y " % ( dirname, len(f_x), len(f_y) )

    if len( f_x ) == len( f_y ):
        print "all ok!"
        return
    elif len(f_x) > len(f_y):
        longest = f_x
        shortest = f_y
        arg = '_y_'
    else:
        longest = f_y
        shortest = f_x
        arg = '_x_'

    for file in longest:
        num =  get_num(file)
        matches = [x for x in shortest if x.endswith(arg+num+'.npy') ]
        if len(matches) == 0:
            statement = 'rm '+dirname+file
            print ' Extra file: '+file

            #remove the files if flag correctly set
            if remove == 'Yes':
                print statement
                os.system(statement)


def remove_links(folder, label):

    statement = "rm -f %s" % (folder)

    if not os.path.islink(folder) and label == 'invalid':
        os.system(statement)
    elif label == 'all':
        os.system(statement)

    return

def make_array_list(folder):
    '''
    Create a numpy array holding ( file number, array idex ) ntuples to identify
    each patch in the database with an unique address
    '''

    address_list = [] #np.empty(1, dtype=int)

    for file in os.listdir(folder):
        if '_x_' in file:

            print 'reading '+file

            num = get_num(file)
            try:
                x = np.load(folder+file, mmap_mode='r')
            except:
                print "can't open file"
                continue

            for id in range(0,len(x)):
                address_list.append( (num, id) )

    np.save(folder+"/address_list.npy", address_list)

    return


def main():

    folder="/eos/user/a/ascarpel/CNN/neutrino/numpy/"
    inputdir="2/"
    training_size = int(sys.argv[1])
    testing_size = int(sys.argv[2])
    first_file =  int(sys.argv[3])
    last_file = int(sys.argv[4])

    #do a list of files checking for duplicates
    list = make_file_list( folder+"2/" )

    make_sample_array( training_size, testing_size, folder+inputdir, (first_file, last_file), list )

    #remove dangling symlinks
    remove_links("./training/", "invalid")
    remove_links("./testing/",  "invalid")

    #check the training sample
    checklength( folder+"training/", "Yes" )

    #check the testing sample
    checklength( folder+"testing/", "Yes")

    print "All done"


if __name__ == "__main__":
    main()
