# -*- encoding: utf-8 -*-

""" buidNET.py: Caffe preprocessing buider """

""" If you have a set of images and you want to train a CNN easily, use this tool.

    REQUERIMENTS: You need to have a general folder wich contains many sub-folder, where each one contains your related images.
		e.g. General folder:        IMAGES
		     sub-folders in IMAGES: COW ( inside you have your related pictures: cow1.jpg, cow2.jpg, ... )
					    LION ( inside you have your related pictures: lion1.jpg, lion2.jpg, ... )
					    ...
"""

__author__      = "Pedro Herruzo"
__copyright__   = "Copyright 2015 Pedro Herruzo"

import sys
import os
import getopt
import shutil
import random
### image tools
from skimage import data, draw # not used, you can remove it
from skimage import transform, util, io
from skimage import filters, color # not used, you can remove it
from matplotlib import pyplot as plt # not used, you can remove it


# Target default values
CAFFE_DATA_DIR = '/home/pherrusa7/code/caffe/data'
FOLDER = 'foodCAT' # wich holds all caffe data
TRAIN = 'train.txt'
TEST = 'test.txt'
MAPPING = 'classesID.txt'
IMAGE_HOLDER = 'images' # relative default path where we'll save all images to caffe
SIZE = 256

# Source default values
SOURCE_PATH = '.'
SOURCE_NUM_TRAIN = 200
SOURCE_NUM_VAL = 51





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


def fillNetData(idClass, elements, absClassPath, absdataDirPath):
	'''
		idClass: class id to identify each img in elements
		elements: list of images
		absClassPath: absolute path to the current images holder (class)
		absdataDirPath: target path to copy all images in elements

		retunrs a list of string pairs: absoluteImagePath idClass
	'''

	data = []
	for img in elements:
		# absolute path to current the image
		absFilePath = os.path.join(absClassPath, img)

		# path to caffe image holder
		targetPath = os.path.join(absdataDirPath, img)+'.jpg'

		# Save the image in the caffe image holder
		resizeAndSave(absFilePath, SIZE, SIZE, targetPath)

		# to copy the pic without RESIZE uncomment next line and comment the previous line
		#shutil.copy(absFilePath, absdataDirPath)

		# add the image path and their class to train
		data.append(str(targetPath)+' '+str(idClass))

	return data


def getDirData(path):

	# get the generator of subdirs contents
	dirsData = os.walk(path)

	# left out the frist element, wich contains only the names of subdirectories
	a = dirsData.next()

	dirs = {}

	for (root , dirnames, filenames) in dirsData:
		# from position 2 to remove './'
		dirs[root[2:]] = filenames
		# also we replace white spaces with underline
		#dirs[root[2:].replace (' ', '_')] = [files.replace (' ', '_') for files in filenames]

	#print 'dirs:', dirs
	return dirs


def buildCaffeData(path, numTrain, numVal, dataDir):
	'''
		This function store at 'dataDir' folder the folder FOLDER with the nexts docs:
		IMAGE_HOLDER: folder to hold all the images
		TRAIN: train set file .txt with 'numTrain' numbers of rows, where each one is a pair: fullPathToImage class
		TEST: test set file .txt with 'numVals' numbers of rows, where each one is a pair: fullPathToImage class

		We supose 'path' is the path to your classes holder, as directories, where each directory contains all images
		for this class.
	'''

	# inicialize target lists
	train = []
	test = []
	classMapping = {}

	# absolute path to the current image folder
	absCurrPath = os.path.abspath(path)
	# absolute path to the foodCAT holder
	absTargetPath = os.path.abspath(dataDir)

	# create full path to foodCAT and foodCAT/IMAGE_HOLDER
	foodCATpath = os.path.join(absTargetPath, FOLDER)
	foodCATImgPath = os.path.join(foodCATpath, IMAGE_HOLDER)

	# if exists foodCATpath, delete it with all the contents
	if os.path.exists(foodCATpath):
		shutil.rmtree(foodCATpath)

	# get image names for every folder as {'bacallà': [globalPath/bacalla1.jpg, ...], ...}
	allImage = getDirData(path)

	# create the folders for foodCAT and foodCAT/IMAGE_HOLDER
	os.makedirs(foodCATpath)
	os.makedirs(foodCATImgPath)


	for clas, images in allImage.iteritems():
		# absolute path to the current class
		absClassPath = os.path.join(absCurrPath, clas)

		# add the new class with a new id to 'classMapping'
		idClass = len(classMapping.keys())
		classMapping[idClass] = clas

		# get a list of random images to select random subset of images to train and test
		randomImages = random.sample(images, numTrain+numVal)
		trainSet = randomImages[:numTrain]
		testSet = randomImages[numTrain:numTrain+numVal] #[numTrain:]

		# save the train images in foodCATImgPath and fill train.txt
		train += fillNetData(idClass, trainSet, absClassPath, foodCATImgPath)

		# save the test images in foodCATImgPath and fill test.txt
		test += fillNetData(idClass, testSet, absClassPath, foodCATImgPath)

	# save train and test as .txt in foodCATpath
	saveList2File(os.path.join(foodCATpath, TRAIN), train)
	saveList2File(os.path.join(foodCATpath, TEST), test)
	saveList2File(os.path.join(foodCATpath, MAPPING), classMapping.iteritems())

	print 'mapping: '
	print
	print 'number of train images:', len(train), '. It is ', len(classMapping.keys()), 'classes per', numTrain, 'images.'
	print 'number of test images:', len(test), '. It is ', len(classMapping.keys()), 'classes per', numVal, 'images.'

	# validation
	if ( len(train) != numTrain*len(classMapping.keys()) ) or ( len(test) != numVal*len(classMapping.keys())):
		print 'train and test set validation was incorrect. Mission aborted.'
   		sys.exit(2)

   	print 'everything is gonna be allright...'



def getArgs(argv):

	try:
		opts, args = getopt.getopt(argv, 'p:t:v:d:')
	except getopt.GetoptError:
   		print 'bad usage'
   		sys.exit(2)

   	# Set default values
   	path = SOURCE_PATH
   	numTrain = SOURCE_NUM_TRAIN
  	numVal = SOURCE_NUM_VAL
  	dataDir = CAFFE_DATA_DIR

  	# if they are, get default values
 	for opt, arg in opts:
 		#print 'opt: ',opt, ' arg: ', arg
 		if opt == "-p":
 			path = str(arg)
 		if opt == "-t":
 			numTrain = int(arg)
 		if opt == "-v":
 			numVal = int(arg)
 		if opt == "-d":
 			dataDir = str(arg)

 	# check the arget path
 	if not os.path.exists(dataDir):
 		print 'incorrect path to save all data (-d)'
 		sys.exit(2)

 	return path, numTrain, numVal, dataDir


if __name__=='__main__':
	'''
		-p path wich holds all directories (classes) where each directory has the images
		-t num of images to train
		-v num of images to validate
		-d path to save all data output

		#TODO:
			Hacer que test/val sea un porcentaje del mínimo de imagenes que todas las carpetas tienen
	'''

	# get path and num images of train/val
	path, numTrain, numVal, dataDir = getArgs( sys.argv[1:] )

	# build caffe data
	buildCaffeData(path, numTrain, numVal, dataDir)
