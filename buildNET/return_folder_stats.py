# -*- encoding: utf-8 -*-

""" folder_stats.py: Counts elements of each subdir from a given path
    Example of use: python folder_stats.py -p '/home/pedro/code/my_caffe/handmade_OLD/images'
"""

__author__      = "Pedro Herruzo"
__copyright__   = "Copyright 2015 Pedro Herruzo"


import sys
import os
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

DEFAULT_PATH = '.'

def plot(xy_data, y_mean, y_min, y_max):
    # add mean, min and max to each subdir index
    xy_data['mean'] = y_mean
    xy_data['min'] = y_min
    xy_data['max'] = y_max

    '''
    xy_data['mean'] = pd.Series([y_mean for x in range(len(xy_data.index))], index=xy_data.index)
    xy_data['min'] = pd.Series([y_min for x in range(len(xy_data.index))], index=xy_data.index)
    xy_data['max'] = pd.Series([y_max for x in range(len(xy_data.index))], index=xy_data.index)
    '''

    # Delete column 'food' in orther to print a nice plot (food column only contains names)
    #xy_data.drop('foood', axis=1, inplace=True)

    #print xy_data
    xy_data.plot(kind='bar')

    plt.show()

def computeStats(dict_dirs):

    # get the info as beauty DataFrame, sorted from more to less
    subdirsInfo = pd.DataFrame( (k,len(v)) for k,v in dict_dirs.iteritems() )
    subdirsInfo = subdirsInfo.sort_values(by=1, ascending=False)
    subdirsInfo = subdirsInfo.rename(columns={0: 'foood', 1: 'count'})
    #print subdirsInfo.describe()
    return subdirsInfo, subdirsInfo.describe()



def getDirData(path):

    # get the generator of subdirs contents
    dirsData = os.walk(path)

    # left out the frist element, wich contains only the names of subdirectories
    a = dirsData.next()

    dirs = {}

    for (root , dirnames, filenames) in dirsData:
        dirs[root] = filenames

    return dirs



def getArgs(argv):

    try:
        opts, args = getopt.getopt(argv, 'p:')
    except getopt.GetoptError:
        print 'bad usage'
        sys.exit(2)


    # Set default values
    path = DEFAULT_PATH

    # if they are, get default values
    for opt, arg in opts:
        #print 'opt: ',opt, ' arg: ', arg
        if opt == "-p":
            path = str(arg)

    # check the target path
    if not os.path.exists(path):
        print 'incorrect path to compute the stats (-p)'
        sys.exit(2)

    return path

def descriveFolders( path ):

    # get image names for every folder as {'bacallà': [globalPath/bacalla1.jpg, ...], ...}
    return getDirData(path)


if __name__=='__main__':

    # get path where we compute the stats
    path = getArgs( sys.argv[1:] )
    descriveFolders(path)
