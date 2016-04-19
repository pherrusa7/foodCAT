
import sys
import json

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



def catFoodStatistics( jsonFile_to_load ):

	foods = loadJSON( jsonFile_to_load )

	stat = {}

	for food in foods:

		if food['category'] in stat.keys():			
			stat[ food['category'] ]['numDishes'] += 1

		else:
			stat[ food['category'] ] = { 'numDishes': 1, 'category': food['category'] }

	totDishes = sum( [ cat['numDishes']  for cat in stat.values() ] )

	for cat in stat.values():
		cat['%'] = 100.0*cat['numDishes']/totDishes

	sortedStat = sorted( stat.values(), key = lambda k: k['%'], reverse = True )
	

	return sortedStat


if __name__ == '__main__':

	''' Global Stats
	for cat in catFoodStatistics('foods.json'):
		print cat
	'''

	'''
	for cat in loadJSON('selectedFoods.json'):
		print cat['name']

	print '###########################'
	'''

	# selected foods stats
	for cat in catFoodStatistics('selectedFoods.json'):
		print cat


''' 

 RESULTS

{'category': u'Carnes', 'numDishes': 223, '%': 24.668141592920353}
{'category': u'Pescados y mariscos', 'numDishes': 156, '%': 17.256637168141594}
{'category': u'Postres y dulces', 'numDishes': 123, '%': 13.606194690265486}
{'category': u'Pastas, arroces y otros cereales', 'numDishes': 91, '%': 10.06637168141593}
{'category': u'Verduras y otras hortalizas', 'numDishes': 79, '%': 8.738938053097344}
{'category': u'Sopas, caldos y cremas', 'numDishes': 71, '%': 7.853982300884955}
{'category': u'Huevos', 'numDishes': 46, '%': 5.088495575221239}
{'category': u'Ensaladas y platos frios', 'numDishes': 34, '%': 3.7610619469026547}
{'category': u'Caracoles', 'numDishes': 23, '%': 2.5442477876106193}
{'category': u'Legumbres', 'numDishes': 23, '%': 2.5442477876106193}
{'category': u'Salsas', 'numDishes': 20, '%': 2.2123893805309733}
{'category': u'', 'numDishes': 11, '%': 1.2168141592920354}
{'category': u'Setas', 'numDishes': 4, '%': 0.4424778761061947}



 SELECTED RESULTS

{'category': u'Postres y dulces', 'numDishes': 34, '%': 23.448275862068964}
{'category': u'Carnes', 'numDishes': 27, '%': 18.620689655172413}
{'category': u'Pescados y mariscos', 'numDishes': 26, '%': 17.93103448275862}
{'category': u'Verduras y otras hortalizas', 'numDishes': 11, '%': 7.586206896551724}
{'category': u'Pastas, arroces y otros cereales', 'numDishes': 11, '%': 7.586206896551724}
{'category': u'Sopas, caldos y cremas', 'numDishes': 8, '%': 5.517241379310345}
{'category': u'Legumbres', 'numDishes': 7, '%': 4.827586206896552}
{'category': u'Salsas', 'numDishes': 5, '%': 3.4482758620689653}
{'category': u'Huevos', 'numDishes': 5, '%': 3.4482758620689653}
{'category': u'Ensaladas y platos frios', 'numDishes': 5, '%': 3.4482758620689653}
{'category': u'Caracoles', 'numDishes': 3, '%': 2.0689655172413794}
{'category': u'Setas', 'numDishes': 2, '%': 1.3793103448275863}
{'category': u'', 'numDishes': 1, '%': 0.6896551724137931}


'''

