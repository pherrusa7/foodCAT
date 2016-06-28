for i in range (115,216):
    y_true.append(i)
    y_pred.append(i)

for i in range (0,115):
    y_true.append(i)
    y_pred.append(i)

len(y_pred)==len(y_true)

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from ast import literal_eval

import os

import sys

import getopt

PATH_TO_PROJECT=''
LABELS_FILE = os.path.join(PATH_TO_PROJECT,'foodCAT_resized/classesID.txt')

labels = np.array(np.loadtxt(LABELS_FILE, str, delimiter='\t'))

labelsDICT = dict([(literal_eval(e)[0],literal_eval(e)[1]) for e in labels])

ids = np.array([literal_eval(e)[0] for e in labels])

cm = confusion_matrix(y_true, y_pred)

np.set_printoptions(precision=1)

plt.figure()

caffeWrapper.plot_confusion_matrix(cm,ids)

plt.show()
