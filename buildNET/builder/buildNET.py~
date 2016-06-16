# -*- encoding: utf-8 -*-

""" buidNET.py: Caffe preprocessing buider
                If you have a set of images and you want to train a CNN easily, use this tool.
"""

""" This script will generate the files train, validation and test '.txt' from a given structure of folders with images.

    REQUERIMENTS: You need to have a general folder wich contains many sub-folder, where each one contains your related images.
        e.g. General folder:        IMAGES
             sub-folders in IMAGES: COW ( inside you have your related pictures: cow1.jpg, cow2.jpg, ... )
                                    LION ( inside you have your related pictures: lion1.jpg, lion2.jpg, ... )
                        ...
        Also, you can have more than one general folder, just give the paths to all this general folders that you want to train
        together splited by a comma (without spaces).

    EXAMPLE: python builNET.py -p '/PATH/TO/FIRST/GENERAL/FOLDER','/PATH/TO/SECOND/GENERAL/FOLDER' -t '/PATH/TO/TARGET/FOLDER'

    Where '/PATH/TO/TARGET/FOLDER' is the path where the train, val and test '.txt' will be generated
"""

__author__      = "Pedro Herruzo"
__copyright__   = "Copyright 2015 Pedro Herruzo"

### general inputs
import sys
import os
import getopt
import shutil
import random
### image tools
#from skimage import data, draw # not used, you can remove it
#from skimage import transform, util, io
#from skimage import filters, color # not used, you can remove it
#from matplotlib import pyplot as plt # not used, you can remove it


# Target default values
TARGET_PATH = os.getcwd()
FOLDER = 'foodCAT_resized' # wich holds all caffe data. CHANGE THIS AND WRITE YOUR PROJECT NAME
TRAIN = 'train.txt'
VAL = 'val.txt'
TEST = 'test.txt'
MAPPING = 'classesID.txt'
#IMAGE_HOLDER = 'images' # relative default path where we'll save all images to caffe
#SIZE = 256

# Source default values
SOURCE_PATH = '.'

# Data Values
TRAIN_PERCENTATGE = 0.8
VAL_PERCENTATGE = 0.1
TEST_PERCENTATGE = 0.1
MIN_IMAGES = 100

# Not used
def resizeAndSave(sourceImage, newWidth, newHeight, destName):

    img = io.imread(sourceImage)
    imgResize = transform.resize(img, (SIZE, SIZE))
    #img = img.resize((newWidth, newHeight), Image.ANTIALIAS)
    #print destName
    io.imsave(destName, imgResize)


def saveList2File(path, holder):
    try:
        outfile = open(path, "w")
        print >> outfile, '\n'.join(str(i) for i in holder)
        outfile.close()
    except:
        print 'Cannot save ', path
        sys.exit(2)


def fillNetData(idClass, elements, absClassPath): #, abstargetPathPath):
    '''
        idClass: class id to identify each img in elements
        elements: list of images
        absClassPath: absolute path to the current images holder (class)
        (OPT) abstargetPathPath: target path to copy all images in elements

        retunrs a list of string pairs: absoluteImagePath idClass
    '''

    data = []
    for img in elements:
        # absolute path to current the image
        absFilePath = os.path.join(absClassPath, img)

	'''
        # path to caffe image holder
        targetPath = os.path.join(abstargetPath, img)+'.jpg'

        # Save the image in the caffe image holder
        resizeAndSave(absFilePath, SIZE, SIZE, targetPath)

        # to copy the pic without RESIZE uncomment next line and comment the previous line
        #shutil.copy(absFilePath, abstargetPathPath)
    '''

        # add the image path and their class to train
        data.append(str(absFilePath)+' '+str(idClass))

    return data


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


def buildCaffeData(paths, targetPath):
    '''
        This function store at 'targetPath' folder the folder FOLDER with the nexts docs:
        (OPT) IMAGE_HOLDER: folder to hold all the images
        TRAIN: train file .txt, where each one is a pair: full/Path/To/Image classID
        VAL: val file .txt, where each one is a pair: full/Path/To/Image classID
        TEST: test file .txt, where each one is a pair: full/Path/To/Image classID

        We supose 'path' is a list of paths to your classes holders, as directories, where each directory contains
	all images for this class.
    '''

    # inicialize target lists
    train = []
    val = []
    test = []
    classMapping = {}
    totalClasses = 0
    totalImages = 0

    for path in paths:
        # absolute path to the current image folder
        absCurrPath = os.path.abspath(path)
        # absolute path to the target holder
        absTargetPath = os.path.abspath(targetPath)

        # create full path to foodCAT
        foodCATpath = os.path.join(absTargetPath, FOLDER)

        # if exists foodCATpath, delete it with all the contents
        if os.path.exists(foodCATpath):
            shutil.rmtree(foodCATpath)

        # get image names for every folder as {'globalPath/bacalla': [bacalla1.jpg, ...], ...}
        allImage = getDirData(path)

        # create the folders for foodCAT
        os.makedirs(foodCATpath)
        #print 'should be created the dir: ', foodCATpath

        # Fill the train/val/test txt's
        for clas, images in allImage.iteritems():
            # We use classes which at least have MIN_IMAGES items
            if len(images) >= MIN_IMAGES:

                # absolute path to the current class
                #absClassPath = os.path.join(absCurrPath, clas)

                # add the new class with a new id to 'classMapping'
                idClass = len(classMapping.keys())
                classMapping[idClass] = clas.split('/')[-1] # to get just the class name from the full path

                # Calculate the number of images for each testSet
                numTrain = int(round( len(images)*TRAIN_PERCENTATGE ))
                numVal = int(round( len(images)*VAL_PERCENTATGE ))
                numTest = int(round( len(images)*TEST_PERCENTATGE ))

                # UNCOMMENT NEXT LINE TO get a list of random images for each set
                #random.shuffle(images)
                trainSet = images[:numTrain]
                valSet = images[numTrain:numTrain+numVal] #[numTrain:]
                testSet = images[numTrain+numVal:] #[numTrain:]

                # fill train.txt
                train += fillNetData(idClass, trainSet, os.path.abspath(clas))

                # fill val.txt
                val += fillNetData(idClass, valSet, os.path.abspath(clas))

                # fill test.txt
                test += fillNetData(idClass, testSet, os.path.abspath(clas))

                # for validation
                totalClasses += 1
                totalImages += len(images)

    # save train and test as .txt in foodCATpath
    saveList2File(os.path.join(foodCATpath, TRAIN), train)
    saveList2File(os.path.join(foodCATpath, VAL), val)
    saveList2File(os.path.join(foodCATpath, TEST), test)
    saveList2File(os.path.join(foodCATpath, MAPPING), classMapping.iteritems())

    print 'number of train images:', len(train)
    print 'number of val images:', len(val)
    print 'number of test images:', len(test)
    print 'total classes: ', totalClasses
    print 'total images: ', len(train)+len(val)+len(test)


    if ( totalImages != len(train)+len(val)+len(test) ):
        print 'train,val or test validation was incorrect. Mission aborted.'
        sys.exit(2)

    print 'everything is gonna be allright...'



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


def build( paths, targetPath ):

    print 'sourcePaths: ', paths
    print 'targetPath: ', targetPath

    # build caffe data
    buildCaffeData(paths, targetPath)


if __name__=='__main__':
    '''
        -p path wich holds all directories (classes) where each directory has the images
        -t num of images to train
        -v num of images to validate
        -d path to save all data output

    '''

    # get paths of superClasses*, num images of train/val and target path to hold all files
    paths, targetPath = getArgs( sys.argv[1:] )

    build(paths, targetPath)
