import sys
import os
import getopt
from scipy import misc
import numpy as np
import pandas as pd

SOURCE_PATH = '.'
TARGET_PATH = os.getcwd()
BALANCED = False # CHANGE TO 'True' IF YOU ARE USING BALANCED DATASET)
MIN_IMAGES = 100
MAX_IMAGES = 500  # set as you need IF YOU ARE USING BALANCED DATASET
import pdb

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
#  data = plotPoints.plotPoints(['data/images','data/food-101/images'], '/home/pedro/TFG')
def plotPoints(paths, targetPath):
    '''
    '''

    # data will contain the list of each dataset you are using
    data = {}

    # absolute path to the target holder
    absTargetPath = os.path.abspath(targetPath)


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


    y = pd.DataFrame(data['data/food-101/images']['points'])[0]
    x = pd.DataFrame(data['data/food-101/images']['points'])[1]
    plt.scatter(x, y, alpha=.1, s=400)

    plt.scatter(x, y, alpha=.1, s=400)
    x1 = pd.DataFrame(data['data/images']['points'])[1]
    y1 = pd.DataFrame(data['data/images']['points'])[0]
    plt.scatter(x1, y1, alpha=.1, s=400, color='red')

    plt.show()

    return data

def getArgs(argv):

    try:
        opts, args = getopt.getopt(argv, 'p:t:')
    except getopt.GetoptError:
        print 'bad usage'
        sys.exit(2)

    # Set default values
    path = SOURCE_PATH
    targetPath = TARGET_PATH

    # if they are, get default values
    for opt, arg in opts:
        #print 'opt: ',opt, ' arg: ', arg
        if opt == "-p":
            path = str(arg)
        if opt == "-t":
            targetPath = str(arg)

    # check the target path
    if not os.path.exists(targetPath):
        print 'incorrect path to save all data (-d)'
        sys.exit(2)

    # check the sources path
    for sourcePath in path.split(","):
        if not os.path.exists(sourcePath):
            print 'incorrect source path of classes'
            sys.exit(2)


    return path.split(","), targetPath


#run plotPoints.py -p 'data/images','data/food-101/images'
if __name__=='__main__':
    '''
        -p path wich holds all directories (classes) where each directory has the images
        -t where to save the plot

    '''

    # get paths of superClasses*, num images of train/val and target path to hold all files
    paths, targetPath = getArgs( sys.argv[1:] )

    plotPoints(paths, targetPath)
