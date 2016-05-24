"""
usage from TFG folder:  python tools/scripts/caffeWrapper.py

python tools/scripts/caffeWrapper.py foodCAT_googlenet_food101
python tools/scripts/caffeWrapper.py foodCAT_alexnet
python tools/scripts/caffeWrapper.py foodCAT_VGG_ILSVRC_19_layers
"""

import os
import sys
import caffe
#import pdb; pdb.set_trace()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ast import literal_eval

PATH_TO_PROJECT=''
TEST = os.path.join(PATH_TO_PROJECT,'foodCAT/test.txt')
TEST_just_foodCAT = os.path.join(PATH_TO_PROJECT,'foodCAT/test_just_foodCAT.txt')
LABELS_FILE = os.path.join(PATH_TO_PROJECT,'foodCAT/classesID.txt')

models  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/snapshots/bvlc_alexnet.caffemodel"),
	   "foodCAT_googlenet_food101":	  os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/snapshots/first_TRAIN_73_91/ss_foodCAT_googlenet_food101_train_iter_490000.caffemodel"),
	   "foodCAT_VGG_ILSVRC_19_layers": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/snapshots/ss_foodCAT_VGG_ILSVRC_19_layers_train_iter_80000.caffemodel")}

# The solvers itselfs points to the network configuration (to TRAIN and VAL)
solvers  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/solver.prototxt"),
	   "foodCAT_googlenet_food101":	  os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/solver.prototxt"),
	   "foodCAT_VGG_ILSVRC_19_layers": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/CLUSTER/solver.prototxt")}

"""
    To be as general as posible script, we need to fit some args before execute:
modelName: Name of the model to use, in order to get the '.caffemodel' file (see definition of dict 'models' for more understanding)
modelType: In order to get the model definition depending on which dataset we'll use (posible options so far, net_TEST or net_TEST_just_foodCAT, where both are dicts on this code)
nameProbsLayer: The top name of the last innerProduct blob used in the model definition
nameAccuracyTop1: The top name of the Accuracy blob used in the model definition
nameAccuracyTop5: The top name of the Accuracy blob used in the model definition with the parameter 'top_k: 5'
"""

################# TEST DEFINITIONS #################

# Here you need to add all fields for a new model to deploy/test
allModels = {"foodCAT_alexnet":
                            {"caffemodel": os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/snapshots/bvlc_alexnet.caffemodel"),
                            "netDefinition":
                                            {"net_TEST": os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
                                            "net_TEST_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO ")},
                            "nameLayer_AccuracyTop1": 'TODO',
                            "nameAccuracyTop5": 'TODO',
                            "nameLayer_innerProduct": 'TODO'},
        "foodCAT_googlenet_food101":
                            {"caffemodel": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/snapshots/first_TRAIN_73_91/ss_foodCAT_googlenet_food101_train_iter_490000.caffemodel"),
                            "netDefinition":
                                            {"net_TEST": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/test.prototxt"),
                                            "net_TEST_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/test_just_foodCAT.prototxt")},
                            "nameLayer_AccuracyTop1": 'loss3/top-1',
                            "nameLayer_AccuracyTop5": 'loss3/top-5',
                            "nameLayer_innerProduct": 'loss3/classifier_foodCAT_googlenet_food101'},
        "foodCAT_VGG_ILSVRC_19_layers":
                            {"caffemodel": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/snapshots/ss_foodCAT_VGG_ILSVRC_19_layers_train_iter_80000.caffemodel"),
                            "netDefinition":
                                            {"net_TEST": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/test.prototxt"),
                                            "net_TEST_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/test_just_foodCAT.prototxt")},
                            "nameLayer_AccuracyTop1": 'accuracy@1',
                            "nameLayer_AccuracyTop5": 'accuracy@5',
                            "nameLayer_innerProduct": 'fc8_foodCAT'} }

# Here you need to fill all fields if you want to use another dateset (also you will need to fill the 'netDefinition' for each element in allModels dict)
allDatasets = {"net_TEST":
                            {"numImages": 14630,
                            "numClasses": 218},
            "net_TEST_just_foodCAT":
                            {"numImages": 4530,
                            "numClasses": 117} }


####################### TEST ZONE
# We use the same file where we define the net and replace the TRAIN and VAL data layer by the TEST data layer
# i.e. Pointing to the test.txt in the data layer
# TODO Just use the same net from solvers replacing the TRAIN and VAL data layer by the TEST data layer (in a pycaffe way)

# Net pointing to the TEST data set
net_TEST  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
	   "foodCAT_googlenet_food101":	  os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/test.prototxt"),
	   "foodCAT_VGG_ILSVRC_19_layers": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/test.prototxt")}

# Net pointing to the TEST_just_foodCAT data set constrained just to the Catalan food
net_TEST_just_foodCAT  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
	   "foodCAT_googlenet_food101":	  os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/test_just_foodCAT.prototxt"),
	   "foodCAT_VGG_ILSVRC_19_layers": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/test_just_foodCAT.prototxt")}

diff_TEST_types = {'net_TEST':net_TEST, 'net_TEST_just_foodCAT':net_TEST_just_foodCAT}

# TODO this should be calculated automatically from the net
numImages = {'net_TEST':14630,'net_TEST_just_foodCAT':4530}
numClass = {'net_TEST':218,'net_TEST_just_foodCAT':117}
####################### END TEST ZONE




####################### DEPLOY, to use it as the other people. REQUIRE IMAGE PREPROCESSING AS CAFFE DOES IN TRAIN. (OR NOT???)
deploy  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
	   "foodCAT_googlenet_food101":	  os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/deploy.prototxt"),
	   "foodCAT_VGG_ILSVRC_19_layers": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/CLUSTER/VGG_ILSVRC_19_layers_deploy.prototxt")}

mean  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
	   "foodCAT_googlenet_food101":	  [104, 117, 123],
	   "foodCAT_VGG_ILSVRC_19_layers": [104, 117, 123]}
####################### END DEPLOY


def intNET_TEST_mode( model_def, model_weights, gpu=True ):
    """ model_def is your test.prototxt
        model_weights is your model.caffemodel
    """

    if gpu:
        # Set Caffe to GPU
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    # Create the net is TEST mode
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    return net

# NOT USED
def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    '''
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    '''
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def accuracy_predictions_groundTruth(net, num_batches, batch_size, numClasses_to_predict, nameProbsLayer, nameAccuracyTop1, nameAccuracyTop5):
    """
    """

    ### Set variables
    # We get the number of classes by the size of the vector of the first prediction
    # We can't use it, since when we are applying in just a subset of classes, we still have all classes output probabilities
    numClasses = len(net.blobs[nameProbsLayer].data[0])
    # Inicialize the accuracy reported by the net
    acc_top1 = 0
    acc_top5 = 0
    # Initialize the array to save the accuracy of all our examples
    acc_norm_top1 = [[] for i in xrange(numClasses)]
    # Initialize the arrays for the Confusio Matrix
    y_true = []
    y_pred = []

    # Classify and save data to calculate accuracy and later the confusion matrix
    for t in range(num_batches):
        # Run the net for the current image batch
        net.forward()

        # Update accuracy with the average accuracy of the current batch
        acc_top1 = acc_top1+net.blobs[nameAccuracyTop1].data
        acc_top5 = acc_top5+net.blobs[nameAccuracyTop5].data
        print t

        # Update the class normalized accuracy
        labels = net.blobs['label'].data
        predicted_labels = np.argmax(net.blobs[nameProbsLayer].data, axis=1)
        for pred, lbl in zip(predicted_labels, labels):
            acc_norm_top1[int(lbl)].append(float(pred==lbl))
            y_true.append(lbl)
            y_pred.append(pred)

    # Calculate and print final results
    acc_top1 = acc_top1/num_batches
    acc_top5 = acc_top5/num_batches
    num_classes_with_predictions = sum([1 for e in acc_norm_top1 if e])
    print 'acc_top1: ', acc_top1
    print 'acc_top5: ', acc_top5
    print 'acc_norm_top1: ', sum( [sum(classAccur)/len(classAccur) for classAccur in acc_norm_top1 if len(classAccur)>0] )/num_classes_with_predictions
    print '#classes Total: ', numClasses
    print '#classes to test: ', numClasses_to_predict
    print '#classes with predictions: ', num_classes_with_predictions
    print '#images tested: ', sum([len(classAccur) for classAccur in acc_norm_top1])
    print '#images supose to test: ', num_batches*batch_size

    return y_true, y_pred


##### USAGE EXAMPLE (from ipython)
# import caffeWrapper

# y_true, y_pred  = caffeWrapper.customTEST('foodCAT_googlenet_food101', 'net_TEST')
# y_true, y_pred = caffeWrapper.customTEST('foodCAT_googlenet_food101', 'net_TEST_just_foodCAT')

# y_true, y_pred = caffeWrapper.customTEST('foodCAT_VGG_ILSVRC_19_layers', 'net_TEST')
# y_true, y_pred = caffeWrapper.customTEST('foodCAT_VGG_ILSVRC_19_layers', 'net_TEST_just_foodCAT')
def customTEST(model, dataset):
    """
    model: Name of the architecture we will use
    dataset: Name of the dataset we will use for the test (note that this field is Required because we may use diferents test.prototxt for one architecture,
                                                           where the difference between them lay just in the dataset that they are pointing)
             So far it has 2 options: 'net_TEST' or "net_TEST_just_foodCAT"
    """

    # Init the net
    model_weights = allModels[model]['caffemodel']
    model_def = allModels[model]['netDefinition'][dataset]
    net = intNET_TEST_mode(model_def, model_weights)

    # Get attributes and print some info
    numImages = allDatasets[dataset]['numImages']               # number of image on the current dataset
    numClasses_to_predict = allDatasets[dataset]['numClasses']  # number of classes we suposse to predict
    batch_size = net.blobs['data'].num                          # batch size for each forward of the net
    num_batches = int(numImages/batch_size)                     # Ammount of batches required to test all images in the current dataset
    nameProbsLayer = allModels[model]["nameLayer_innerProduct"]
    nameAccuracyTop1 = allModels[model]["nameLayer_AccuracyTop1"]
    nameAccuracyTop5 = allModels[model]["nameLayer_AccuracyTop5"]
    print 'Total number of images to Test: ', numImages, ' in batches of: ', batch_size
    print 'Required Iterations: ', num_batches

    # Get the predictions and the ground truth. Also print the custom accuracy
    y_true, y_pred = accuracy_predictions_groundTruth(net, num_batches, batch_size, numClasses_to_predict, nameProbsLayer, nameAccuracyTop1, nameAccuracyTop5)

    # Caluculate and plot the confusion matrix
    labels = np.array(np.loadtxt(LABELS_FILE, str, delimiter='\t'))
    ids = np.array([literal_eval(e)[0] for e in labels])
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, ids)

    # Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, ids, title='Normalized confusion matrix')

    plt.show()


if __name__ == "__main__":

    net = deployModel(sys.argv[1])
