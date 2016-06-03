
import os
import sys
import caffe
#import pdb; pdb.set_trace()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from ast import literal_eval

PATH_TO_PROJECT=''
'''
TEST = os.path.join(PATH_TO_PROJECT,'foodCAT/test.txt')
TEST_just_foodCAT = os.path.join(PATH_TO_PROJECT,'foodCAT/test_just_foodCAT.txt')
'''

# THIS DEPENDS ON THE DATASET THAT YOU ARE USING
LABELS_FILE = os.path.join(PATH_TO_PROJECT,'foodCAT_resized/classesID.txt')
labels = np.array(np.loadtxt(LABELS_FILE, str, delimiter='\t'))
labelsDICT = dict([(literal_eval(e)[0],literal_eval(e)[1]) for e in labels])



################# TEST DEFINITIONS #################

# Here you need to add all fields for a new model to deploy/test
allModels = {"foodCAT_alexnet":
                            {"caffemodel": os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/snapshots/bvlc_alexnet.caffemodel"),
                            "netDefinition":
                                            {"net_TEST": os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
                                            "net_TEST_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO ")},
                            "nameLayer_AccuracyTop1": 'TODO',
                            "nameLayer_AccuracyTop5": 'TODO',
                            "nameLayer_innerProduct": 'TODO',
                            "solver": os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/solver.prototxt")}, # solver is not used to TEST
        "foodCAT_googlenet_food101":
                            {"caffemodel": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/snapshots/first_TRAIN_73_91/ss_foodCAT_googlenet_food101_train_iter_490000.caffemodel"),
                            "netDefinition":
                                            {"net_TEST": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/test.prototxt"),
                                            "net_TEST_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/test_just_foodCAT.prototxt")},
                            "nameLayer_AccuracyTop1": 'loss3/top-1',
                            "nameLayer_AccuracyTop5": 'loss3/top-5',
                            "nameLayer_innerProduct": 'loss3/classifier_foodCAT_googlenet_food101',
                            "solver": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/solver.prototxt")}, # solver is not used to TEST
        "foodCAT_VGG_ILSVRC_19_layers":
                            {"caffemodel": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/snapshots/ss_foodCAT_VGG_ILSVRC_19_layers_train_iter_80000.caffemodel"),
                            "netDefinition":
                                            {"net_TEST": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/test.prototxt"),
                                            "net_TEST_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/test_just_foodCAT.prototxt")},
                            "nameLayer_AccuracyTop1": 'accuracy@1',
                            "nameLayer_AccuracyTop5": 'accuracy@5',
                            "nameLayer_innerProduct": 'fc8_foodCAT',
                            "solver": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/CLUSTER/solver.prototxt")}, # solver is not used to TEST
        "foodCAT_googlenet_food101_500":
                            {"caffemodel": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101_500/snapshots/ss_foodCAT_googlenet_food101_500_iter_275000.caffemodel"),
                            "netDefinition":
                                            {"net_TEST_balanced": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101_500/test.prototxt"),
                                            "net_TEST_balanced_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101_500/test_just_foodCAT.prototxt"),
                                            "net_TEST": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101_500/test_OLD.prototxt"),
                                            "net_TEST_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101_500/test_just_foodCAT_OLD.prototxt")},
                            "nameLayer_AccuracyTop1": 'loss3/top-1',
                            "nameLayer_AccuracyTop5": 'loss3/top-5',
                            "nameLayer_innerProduct": 'loss3/classifier_foodCAT_food101_500',
                            "solver": os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101_500/solver.prototxt")}, # solver is not used to TEST
        "foodCAT_VGG_ILSVRC_19_layers_500":
                            {"caffemodel": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers_500/snapshots/ss_foodCAT_VGG_ILSVRC_19_layers_500_iter_40000.caffemodel"),
                            "netDefinition":
                                            {"net_TEST_balanced": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers_500/test.prototxt"),
                                            "net_TEST_balanced_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers_500/test_just_foodCAT.prototxt"),
                                            "net_TEST": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers_500/test_OLD.prototxt"),
                                            "net_TEST_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers_500/test_just_foodCAT_OLD.prototxt")},
                            "nameLayer_AccuracyTop1": 'accuracy@1',
                            "nameLayer_AccuracyTop5": 'accuracy@5',
                            "nameLayer_innerProduct": 'fc8_foodCAT_500',
                            "solver": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers_500/solver.prototxt")}, # solver is not used to TEST
        "googlenet_resized":
                            {"caffemodel": os.path.join(PATH_TO_PROJECT, "models/googlenet_resized/snapshots/ss_googlenet_resized_iter_157992.caffemodel"),
                            "netDefinition":
                                            {"net_TEST_resized": os.path.join(PATH_TO_PROJECT, "models/googlenet_resized/test.prototxt"),
                                            "net_TEST_resized_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/googlenet_resized/test_just_foodCAT.prototxt")},
                            "nameLayer_AccuracyTop1": 'loss3/top-1',
                            "nameLayer_AccuracyTop5": 'loss3/top-5',
                            "nameLayer_innerProduct": 'loss3/classifier_resized',
                            "solver": os.path.join(PATH_TO_PROJECT, "models/googlenet_resized/solver.prototxt")}, # solver is not used to TEST
        "googlenet_resized_balanced":
                            {"caffemodel": os.path.join(PATH_TO_PROJECT, "models/googlenet_resized_balanced/snapshots/ss_googlenet_resized_balanced_iter_27360.caffemodel"),
                            "netDefinition":
                                            {"net_TEST_balanced": os.path.join(PATH_TO_PROJECT, "models/googlenet_resized_balanced/test.prototxt"),
                                            "net_TEST_balanced_just_foodCAT": os.path.join(PATH_TO_PROJECT, "models/googlenet_resized_balanced/test_just_foodCAT.prototxt")},
                            "nameLayer_AccuracyTop1": 'loss3/top-1',
                            "nameLayer_AccuracyTop5": 'loss3/top-5',
                            "nameLayer_innerProduct": 'loss3/classifier_resized',
                            "solver": os.path.join(PATH_TO_PROJECT, "models/googlenet_resized_balanced/solver.prototxt")} } # solver is not used to TEST

# Here you need to fill all fields if you want to use another dateset (also you will need to fill the 'netDefinition' for each element in allModels dict)
# TODO: Actually numImages we can read it automatically from the net (HOW?), and numClasses could be just the parameter 'num_classes_with_predictions', as it's calculated
# with the ground True labels.
allDatasets = {"net_TEST":
                            {"numImages": 14630,
                            "numClasses": 218},
            "net_TEST_just_foodCAT":
                            {"numImages": 4530,
                            "numClasses": 117},
            "net_TEST_balanced":
                            {"numImages": 9124,
                            "numClasses": 216},
            "net_TEST_balanced_just_foodCAT":
                            {"numImages": 4074,
                            "numClasses": 115},
            "net_TEST_resized":
                            {"numImages": 14516,
                            "numClasses": 216},
            "net_TEST_resized_just_foodCAT":
                            {"numImages": 4416,
                            "numClasses": 115}}



####################### [not used] DEPLOY, to use it as the other people. REQUIRE IMAGE PREPROCESSING AS CAFFE DOES IN TRAIN. (OR NOT???)
deploy  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
	   "foodCAT_googlenet_food101":	  os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/deploy.prototxt"),
	   "foodCAT_VGG_ILSVRC_19_layers": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/CLUSTER/VGG_ILSVRC_19_layers_deploy.prototxt")}

mean  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
	   "foodCAT_googlenet_food101":	  [104, 117, 123],
	   "foodCAT_VGG_ILSVRC_19_layers": [104, 117, 123]}
####################### END DEPLOY



################# TEST FUNCTIONS #################

def intNET_TEST_mode( model_def, model_weights, gpu=True ):
    """ model_def: is your test.prototxt
        model_weights: is your model.caffemodel
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
    """ cm: a Confusion matrix
        lables: a np.array which contains in each position your labels
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def accuracy_predictions_groundTruth(net, num_batches, batch_size, numClasses_to_predict, nameProbsLayer, nameAccuracyTop1, nameAccuracyTop5):
    """ net: The net to produce the predictions
        num_batches: num iterations of the forward pass
        batch_size: ammount of images in each iteration (forward pass)
        numClasses_to_predict: desired number of classes that the network should predict (to compute a comparation)
        nameProbsLayer: layer name of the last inner product of your net (which gives the probability to pertain to each of your possibles class)
        nameAccuracyTop1: layer name of the accuracy top-1 blob
        nameAccuracyTop5: layer name of the accuracy top-5 blob

        obs: The names of the accuracy layers and probs are needed in order to allow the script get the desired ouputs.
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
            if lbl>=216:
                print 'big: ', lbl
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

################################## second models
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_googlenet_food101_500', 'net_TEST')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_googlenet_food101_500', 'net_TEST_just_foodCAT')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_googlenet_food101_500', 'net_TEST_balanced')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_googlenet_food101_500', 'net_TEST_balanced_just_foodCAT')

# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_VGG_ILSVRC_19_layers_500', 'net_TEST')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_VGG_ILSVRC_19_layers_500', 'net_TEST_just_foodCAT')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_VGG_ILSVRC_19_layers_500', 'net_TEST_balanced')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_VGG_ILSVRC_19_layers_500', 'net_TEST_balanced_just_foodCAT')
################################## first models
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_googlenet_food101', 'net_TEST')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_googlenet_food101', 'net_TEST_just_foodCAT')

# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_VGG_ILSVRC_19_layers', 'net_TEST')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.customTEST('foodCAT_VGG_ILSVRC_19_layers', 'net_TEST_just_foodCAT')
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
    ids = np.array([literal_eval(e)[0] for e in labels])
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, ids)

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, ids, title='Normalized confusion matrix')

    # Show both confusion matrix
    plt.show()

    return y_true, y_pred, cm, cm_normalized



def worst_K_classified(cm, k=6):
    # sorted maxArg matrix reverse (biiger first) (by the option '-' before the cm)
    maxargs = np.argsort(-cm)
    # take the k first
    k_maxargs = maxargs[:,:k]

    # build the structure to see the labels for each classified



################################## resized models
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('googlenet_resized', 'net_TEST_resized')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('googlenet_resized', 'net_TEST_resized_just_foodCAT')

# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('googlenet_resized_balanced', 'net_TEST_balanced')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('googlenet_resized_balanced', 'net_TEST_balanced_just_foodCAT')
################################## second models
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_googlenet_food101_500', 'net_TEST')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_googlenet_food101_500', 'net_TEST_just_foodCAT')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_googlenet_food101_500', 'net_TEST_balanced')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_googlenet_food101_500', 'net_TEST_balanced_just_foodCAT')

# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_VGG_ILSVRC_19_layers_500', 'net_TEST')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_VGG_ILSVRC_19_layers_500', 'net_TEST_just_foodCAT')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_VGG_ILSVRC_19_layers_500', 'net_TEST_balanced')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_VGG_ILSVRC_19_layers_500', 'net_TEST_balanced_just_foodCAT')
################################## first models
#y_true, y_pred, cm, cm_normalized = fastTEST()
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_googlenet_food101', 'net_TEST')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_googlenet_food101', 'net_TEST_just_foodCAT')

# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_VGG_ILSVRC_19_layers', 'net_TEST')
# y_true, y_pred, cm, cm_normalized  = caffeWrapper.fastTEST('foodCAT_VGG_ILSVRC_19_layers', 'net_TEST_just_foodCAT')
def fastTEST(model, dataset):

    y_true, y_pred, cm, cm_normalized  = customTEST(model, dataset)
    diff =0
    eq =0

    for t,p in zip(y_true, y_pred):
        if t==p:
            eq=eq+1
        else:
            diff=diff+1
            if p>114 and t<=114: # 114 are the catalan classes
                print '######## diff'
                print 't: ',t, 'that is ', labelsDICT[t]
                print 'p:', p, 'that is ', labelsDICT[p]

    print 'diff: ', diff
    print 'eq: ', eq
    return y_true, y_pred, cm, cm_normalized


#TODO def lookUP(target_labels, current_net_labels):

def pruebas(y_true, y_pred):

    diff =0
    eq =0

    for t,p in zip(y_true, y_pred):
        if t==p:
            eq=eq+1
        else:
            diff=diff+1
            if p>116:
                print 't: ',t, ' p:', p, ' DIFF'

    print 'diff: ', diff
    print 'eq: ', eq



if __name__ == "__main__":

    net = deployModel(sys.argv[1])
