# -*- coding: latin-1 -*-
""" This script finds Imges by a given names """


import json # To work with JSON format
import sys # To get out of the APP if something is wrong
from googleTranslate import translate # To translate few words
import re  # To clean some strings
from imageRetrieve import getImages # To get images from google Image search  


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




def saveJSON( jsonData, fileName ):
	""" This function save a json object as fileName.json """


	try:
		with open(fileName+'.json', 'w') as fp:
			json.dump(jsonData, fp)
    	
	except:	
		print 'Cannot save'
		sys.exit(1)



def englishTranslate( list_of_dicts ):
	"""
	This funcion recives a list of dicts, where each dict has attributes 'name' and 'category' such values are going to 
	be  translated to English and added as new attributes 'name_E' and 'category_E' in the dict
	"""

	for food in list_of_dicts:

		food['name_E'] = translate( food['name'].encode("utf-8") )
		food['category_E'] = translate( food['category'].encode("utf-8") )

		if food.has_key('name_2'): # If has 'name_2' attribute, it has 'name' attribute
			food['name_E2'] = translate( food['name_2'].encode("utf-8") )

	return list_of_dicts


def cleanFoodName( list_of_dicts ):
	"""
	This function looks for foods who has the name in more than one language and if this match, puts the second 
	name in another attribute.

	If a food has more than ona language, will have two, and the second one will be writted betwen parenthesis. 
	Example: name: huevo frito (ou ferrat)

	Obs: We use raw strings instead of strings to avoid the backslash plague on regex pattern
	"""

	# NOT USED: [\w] matchs any alphanumeric character, equivalent to [a-zA-Z0-9_]  
	# '.' (Dot.) This matches any character except a newline. 
	# '*'  Causes the resulting RE to match 0 or more repetitions of the preceding RE, as many repetitions as 
	#      are possible. ex: ab* will match ‘a’, ‘ab’, or ‘a’ followed by any number of ‘b’s.
	# '?'  Causes the resulting RE to match 0 or 1 repetitions of the preceding RE. ab? will match either ‘a’ or ‘ab’.
	regex = re.compile( r'\(.*?\)' )

	# For each food, check if the name attribute matches the regex
	for food in list_of_dicts:
		foodName = food['name']

		if foodName:
			# Get the sentence associated to the regex (at most is one) or None if does not match
			name_2 = regex.search( foodName )

			if name_2:
				# Delete it from the original attribute
				food['name'] = foodName[:name_2.start()].strip()

				# write as a different attribute without the parenthesis
				food['name_2'] = name_2.group()[1:len(name_2.group())-1]

	return list_of_dicts


def cleanAndTranslate ( jsonFile_to_load, jsonFile_to_save ):
	""" This function load a list of foods dicts as JSON file, clean the data and translates it to English """

	# Load the foods JSON file
	allFood = loadJSON( jsonFile_to_load )

	# Clean the food names
	food = cleanFoodName( allFood )

	# Get English translations for name and category
	foodWithTranslations = englishTranslate( food )

	# Save it as JSON format
	saveJSON( foodWithTranslations, jsonFile_to_save )




def imageSearchOnGoogle ():

	# load the foods name who has already the images downloader
	foods_with_images = loadJSON( 'foods_with_images.json' )

	# read the food dict
	foods = loadJSON( 'selectedFoods.json' )

	# for each food, if it hasn't allready the pictures, download it
	for food in foods:
		if not food['name'] in foods_with_images:
			print
			print '################# ', food['name'], ' #################'

			# download images and get a list with the image names
			images = getImages( food['name'] )
			# If we download images succefully 
			if images:
				# let's add image pairs [URL, name] list to the food
				food['images'] =  { 'googleImages': images }

				# add the food to the list of food_with_images
				foods_with_images.append( food['name'] )
				
				# save everything
				saveJSON( foods_with_images, 'foods_with_images' )
				saveJSON( foods, 'selectedFoodsWIthImages' )
		



if __name__ == "__main__":

	# clean and translate the data OBS: UNCOMMENT NEXT LINE TO USE IT
	#cleanAndTranslate( 'food.json', 'normalized_food' )

	# search Images on google Images
	imageSearchOnGoogle()




