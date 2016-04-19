""" This is a Web scraping program:	Let's scrap 'http://www.cuinacatalana.eu' looking for food names and categories """


import sys
import json
import urllib2
from bs4 import BeautifulSoup as bs




def readHTML( url ):
	""" This function reads and returns the HTML from a URL """

	try:
		return urllib2.urlopen(url).read()
	except:
		print 'Cannot acces to the given URL'
		system.exit(1)




def cleanIngredientsSoup( ingredientsSoup ):
	""" This function removes all dirty in Ingredientes Soup format """

	# break line splited and then delete empty positions of the array
	blSplited = [x.strip() for x in ingredientsSoup.split("\n") if x.strip() != '']

	# return without the title 'Ingredientes'
	return blSplited[1:]



def getFoodProperties( soup ):
	""" This function read and returns the title, the category and the ingredients for a soup object 
		if it has any sense """

	name = soup.find('h1', attrs = {'class' : 'title'}).text 

	# .strip() it's like trim in SQL: remove 'white' spaces
	category = soup.find('div', attrs = {'class' : 'section-wrapper review'}).find('div', attrs = {'class' : 'section'}).text.strip()

	# Get the Ingredients and clean the soup given by the HTML
	ingredients = cleanIngredientsSoup( soup.find('div', attrs = {'class' : 'right-panel'}).find('div', attrs = {'class' : 'inner'}).text )  

	return category, name, ingredients




def saveJSON( data, fileName ):
	""" This function save a json object as fileName.json """

	try:
		with open(fileName+'.json', 'w') as fp:
			json.dump(data, fp)
    	
	except:	
		print 'Cannot save'
		sys.exit(1)



if __name__ == "__main__":

	URL = 'http://www.cuinacatalana.eu/es/pag/receptes/?id='

	N = 2000

	foods = {}

	for i in range(1,N+1):
		# get the html from the dynamic URL and transform it to soup format
		soup = bs( readHTML( URL+str( i ) ) )

		# Print the entired HTML
		#print(soup.prettify().encode('utf-8'))#.strip())

		# get category, name and ingredients of this food
		category, name, ingredients = getFoodProperties( soup )

		# add this element to the foods dict. If already exists, replace it
		foods[i] = {'name' : name, 'category' : category, 'ingredients' : ingredients}

		# print category
		# print name
		# print
	
	# order the values by category ¿ and then by name  ? seems like it does  
	orderedList = sorted(foods.values(), key = lambda k: k['category'] )
	# work but is not what we expect: orderedList = sorted(foods.values(), key = lambda k: ( ['category'], k['name']) )
	# if we want to do it with the keys too: orderedList = sorted(foods.items(), key = lambda k: ( k[1] ) ) #k['category'], k['name']) )

	# save in json format
	saveJSON(orderedList, 'foodsTest') # main file is food.json


	
	
