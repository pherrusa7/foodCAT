""" imageRetrieve.py: Google Image Scrapper """

__author__      = "Pedro Herruzo"
__copyright__   = "Copyright 2015 Pedro Herruzo"

from bs4 import BeautifulSoup

import os
import random
import requests
import urllib2

def getRandomAgent():
    '''
        Returns a random User-Agent.
    '''

    userAgents = [
        {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/33.0.1750.149 Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:40.0) Gecko/20100101 Firefox/40.0" },
        {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36" },
        {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:39.0) Gecko/20100101 Firefox/39.0" }
    ]

    return userAgents[random.randint(0, len(userAgents)-1)]

def getImageURLs(URL, query, start, session):
    '''
        Requests google for image URLs related to 'query' and returns them.
    '''

    imageURLs = []
    width = 1366
    height = 768

    params = {
        "q": query,
        "sa": "X",
        "biw": width,
        "bih": height,
        "tbm": "isch",
        "ijn": start/100,
        "start": start
    }

    request = session.get(URL, params=params)
    bs = BeautifulSoup(request.text)

    for img in bs.findAll("div", {"class": "rg_di"}):
        try:
            imageURLs.append(img.find("img").attrs["data-src"])
        except:
            pass

    return imageURLs

def saveImage(image, name, path):
    '''
        Saves the 'image' in the specified 'path/name' as 'name'_i, where i stands for
        the number of images contained in that directory plus one. If 'path/name' does
        not exist it will be created.
    '''

    mainPath = os.path.join(path, name)

    # create new path if necessary
    if not os.path.exists(mainPath):
        os.makedirs(mainPath)

    idx = len([i for i in os.listdir(mainPath) if name in i]) + 1

    nameToSave = name + '_' + str(idx)

    f = open(os.path.join(mainPath, nameToSave), 'wb')

    try:
        f.write(image)
        f.close()
        return nameToSave

    except:
        print "Could not save %s" % nameToSave
        f.close()
        return None

def downloadImage(imageURL):
    '''
        Downloads and returns the image corresponding to 'imageURL'.
    '''

    try:
        return urllib2.urlopen(imageURL).read()

    except: 
        print "Could not download image \'%s\'" % imageURL
        return None

def downloadAndSaveImages(path, query, imageURLs):
    '''
        Downloads the images corresponding to each URL from 'imageURLs' into 'path/query' directory.
        Both downloading and saving the images is done sequentially, to force the system spend
        more time between queries and (try to) avoid getting banned.
    '''

    downloadedImages = []

    # save the images
    for imgURL in imageURLs:
        rawImg = downloadImage(imgURL)

        if rawImg:
            # save the image in the directory
            imageName = saveImage(rawImg, query, path)

            if imageName:
                # add the image name to the list of downloaded images
                downloadedImages.append( [imgURL, imageName] )

    return downloadedImages

def getImages(imageName, pathToSave="images"):
    '''
        Downloads as much as 1000 pictures related to the google query 'imageName' and saves
        them inside 'pathToSave/imageName'.

        args:
            imageName: image data to search for
            pathToSave: root path where the images will be saved (e.g.: pathToSave/imageName/imageName_i.jpg)

        return: a list of the downloaded image pair [URL, name]
    '''

    URL = "https://www.google.com/search"
    images = []
    imageURLs = []
    s = requests.session()
    N = 10 # Google service limits N to be not greater than 1000

    for x in range(0, N):

        # change the User-Agent on each service call to (try to) avoid getting banned
        s.headers.update(getRandomAgent())

        # request image URLs
        tmpImageURLs = getImageURLs(URL, imageName, 100*x, s)

        # download unique images only
        unqImageURLs = list(set(tmpImageURLs) - set(imageURLs))
        print x, ": ", len(unqImageURLs)

        if unqImageURLs:
            # download the images and append their pairs [URL, name] into 'images'
            images += downloadAndSaveImages(pathToSave, imageName, unqImageURLs)
            imageURLs += unqImageURLs

    return images

if __name__ == "__main__":

    import sys

    print getImages(" ".join(sys.argv[1:]))