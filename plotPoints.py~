import sys
import os
import getopt
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
#import pdb

SOURCE_PATH = '.'
#TARGET_PATH = os.getcwd()
BALANCED = False # CHANGE TO 'True' IF YOU ARE USING BALANCED DATASET)
MIN_IMAGES = 100
MAX_IMAGES = 500  # set as you need IF YOU ARE USING BALANCED DATASET


# data2 = loadJSON( 'data_resolution.json' )
def loadJSON( path ):
	"""	This function load a .json file and return the data contained """


	try:
		#file = open( path, r)
		with open(path , 'r') as fp:
			data = json.load(fp)

	except ValueError:
		print 'Decoding JSON has failed'
		sys.exit(1)

	except:
		print 'Cannot acces to the given path'
		sys.exit(1)

	return data

# saveJSON( data, 'data_resolution' )
def saveJSON( jsonData, fileName ):
	""" This function save a json object as fileName.json """


	try:
		with open(fileName+'.json', 'w') as fp:
			json.dump(jsonData, fp)

	except:
		print 'Cannot save'
		sys.exit(1)


def getImagesSize(clas, images):
    '''
    input looks as [bacalla1.jpg, bacalla2.jpg, ...]
    output are the pair of size for each image on the input list [(225,223), (666,565), ...]
    '''
    sizes = [misc.imread(os.path.join(clas, path)).shape[:2] for path in images]

    return sizes



def getDirData(path):

    # get the generator of subdirs contents
    dirsData = os.walk(path)

    # left out the frist element, wich contains only the names of subdirectories
    a = dirsData.next()

    dirs = {}

    for (root , dirnames, filenames) in dirsData:
        dirs[root] = filenames
        # also we replace white spaces with underline
        #dirs[root[2:].replace (' ', '_')] = [files.replace (' ', '_') for files in filenames]

    #print 'dirs keys:', dirs.keys()
    return dirs

# import plotPoints
#  data = plotPoints.plotPoints(['data/images','data/food-101/resized_images'])
def plotPoints(paths):
    '''
    '''

    # data will contain the list of each dataset you are using
    data = {}

    for path in paths:
        # Set a new list in the data dict
        data[path] = {'points':[] }

        # get image names for every folder as {'globalPath/bacalla': [bacalla1.jpg, ...], ...}
        allImage = getDirData(path)

        # Fill data dict with all image sizes for each class
        for clas, images in allImage.iteritems():
            # We use classes which at least have MIN_IMAGES items
            if len(images) >= MIN_IMAGES:
                # HERE IS THE CHANGE THAT MAKES THE DIFFERENCE WITH buildNET.py
                if BALANCED:
                    images = images[:MAX_IMAGES]

                #
                sizes = getImagesSize(clas, images)
                data[path]['points'].extend(sizes)

        x_max, y_max = np.amax(data[path]['points'], axis=0)
        x_min, y_min = np.amin(data[path]['points'], axis=0)
        data[path]['max_width'] = y_max
        data[path]['min_width'] = y_min
        data[path]['max_hight'] = x_max
        data[path]['min_hight'] = x_min
        print path, ' info: '
        print pd.DataFrame(data[path]['points']).describe()

    saveJSON( data, 'data_resolution_RESIZED' )

    # Plot the pixel resolution of all dataset. CHANGES THIS IF YOU CHANGE THE DATASET PATH
    labels = {'data/food-101/resized_images': 'food-101', 'data/images': 'foodCAT'}
    colors = {'data/food-101/resized_images': 'blue', 'data/images': 'red'}
    plotChart(data, labels, colors)

    return data

def plotChart(data, labels, colors, title='Pixel resolution'):

    plt.title(title)
    plt.ylabel('Height')
    plt.xlabel('Width')

    for dataset in data.keys():
        y = pd.DataFrame(data[dataset]['points'])[0]
        x = pd.DataFrame(data[dataset]['points'])[1]
        plt.scatter(x, y, alpha=.1, s=400, label=labels[dataset], color=colors[dataset])

    plt.legend()
    plt.show()

def plotFromJson():
    ''' This function plots the resolution of de datasets saved at FILE='data_resolution.json'

    USAGE: From the folder container of this script and the file FILE, open ipyhton and type:
    import plotPoints
    data = plotPoints.plotFromJson()
    '''

    data = loadJSON( 'data_resolution_RESIZED.json' )
    labels = {'data/food-101/images': 'food-101', 'data/foodCAT_SR': 'foodCAT'}
    colors = {'data/food-101/images': 'blue', 'data/foodCAT_SR': 'red'}
    plotChart(data, labels, colors)
    return data


def getArgs(argv):

    try:
        opts, args = getopt.getopt(argv, 'p:t:')
    except getopt.GetoptError:
        print 'bad usage'
        sys.exit(2)

    # Set default values
    path = SOURCE_PATH

    # if they are, get default values
    for opt, arg in opts:
        #print 'opt: ',opt, ' arg: ', arg
        if opt == "-p":
            path = str(arg)


    # check the sources path
    for sourcePath in path.split(","):
        if not os.path.exists(sourcePath):
            print 'incorrect source path of classes'
            sys.exit(2)


    return path.split(",")


#run plotPoints.py -p 'data/images','data/food-101/images'
if __name__=='__main__':
    '''
        -p path wich holds all directories (classes) where each directory has the images

    '''

    # get paths of superClasses*, num images of train/val and target path to hold all files
    paths= getArgs( sys.argv[1:] )

    plotPoints(paths)
