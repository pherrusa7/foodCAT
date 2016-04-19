# -*- encoding: utf-8 -*-

""" subDirs_stats.py: Counts elements of each subdir from a given path """

__author__      = "Pedro Herruzo"
__copyright__   = "Copyright 2015 Pedro Herruzo"


import sys
import os
import getopt

DEFAULT_PATH = '.'

def computeStats(dict_dirs):
	stats = {}
	totalFiles = 0

	for folder, list_of_subFiles in dict_dirs.items():
		stats[folder] = len(list_of_subFiles)
		totalFiles += len(list_of_subFiles)

	orderedList = sorted( stats.items(), key = lambda k: (k[1]), reverse=True ) 


	return orderedList, totalFiles

def getDirData(path):

	# get the generator of subdirs contents 
	dirsData = os.walk(path)

	# left out the frist element, wich contains only the names of subdirectories
	a = dirsData.next()

	dirs = {}

	for (root , dirnames, filenames) in dirsData:
		# from position 2 to remove './'
		dirs[root[2:]] = filenames

	return dirs


def getArgs(argv):

	try:
		opts, args = getopt.getopt(argv, 'p')
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

 	# check the arget path
 	if not os.path.exists(path):
 		print 'incorrect path to compute the stats (-d)'
 		sys.exit(2)
 	
 	return path

if __name__=='__main__':
	# get path where we compute the stats
	path = getArgs( sys.argv[1:] )

	# get image names for every folder as {'bacallà': [globalPath/bacalla1.jpg, ...], ...}
	dirData = getDirData(path)

	stats_list, totalFiles = computeStats(dirData)

	for e in stats_list:
		print e

	print 'Number of subDirs: ', len(stats_list)
	print 'Total of items: ', totalFiles
	print 'Median of file by subdirectory: ', 1.0*totalFiles/len(stats_list)