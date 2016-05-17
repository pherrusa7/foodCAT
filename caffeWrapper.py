"""
usage from TFG folder:  python tools/scripts/caffeWrapper.py

python tools/scripts/caffeWrapper.py foodCAT_googlenet_food101
python tools/scripts/caffeWrapper.py foodCAT_alexnet
python tools/scripts/caffeWrapper.py foodCAT_VGG_ILSVRC_19_layers
"""
import os
import sys
import caffe
import pdb;
import numpy as np
from sklearn.metrics import confusion_matrix

PATH_TO_PROJECT=''
TEST=os.path.join(PATH_TO_PROJECT,'foodCAT/test.txt')
TEST_just_foodCAT=os.path.join(PATH_TO_PROJECT,'foodCAT/test_just_foodCAT.txt')

models  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/snapshots/bvlc_alexnet.caffemodel"),
	   "foodCAT_googlenet_food101":	  os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/snapshots/first_TRAIN_73_91/ss_foodCAT_googlenet_food101_train_iter_490000.caffemodel"),
	   "foodCAT_VGG_ILSVRC_19_layers": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/snapshots/ss_foodCAT_VGG_ILSVRC_19_layers_train_iter_80000.caffemodel")}

# The solvers itselfs points to the network configuration (to TRAIN and VAL)
solvers  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/solver.prototxt"),
	   "foodCAT_googlenet_food101":	  os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/solver.prototxt"),
	   "foodCAT_VGG_ILSVRC_19_layers": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/CLUSTER/solver.prototxt")}



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

# NOT USED
def deployModel( modelName, modelType ):
    """ modelType is which TEST dataset are you using: In our example could be 'net_TEST' or 'net_TEST_just_foodCAT'
        modelName is which model we are using: Alexnet, googlenet, VGG_ILSVRC_19_layers, etc.

        returns the net
    """
    print 'Net definition: ', diff_TEST_types[modelType][modelName]
    print 'weights: ', models[modelName]

    # Set Caffe to GPU
    caffe.set_device(0)
    caffe.set_mode_gpu()
    #caffe.set_mode_cpu()

    # Assign net parameters
    model_def = diff_TEST_types[modelType][modelName]
    model_weights = models[modelName]

    # Create the net is TEST mode
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    return net

# NOT USED
def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size, numClasses_to_predict, nameProbsLayer, nameAccuracyTop1, nameAccuracyTop5):
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

    for t in range(num_batches):
        # Run the net for the current image batch
        net.forward()

        # Update accuracy with the average accuracy of the current batch
        acc_top1 = acc_top1+net.blobs[nameAccuracyTop1].data
        acc_top5 = acc_top5+net.blobs[nameAccuracyTop5].data
        print t
        print acc_top1
        #z=[1. for a,b in zip(np.argmax(net.blobs[nameAccuracyTop1].data, axis=1),net.blobs['label'].data) if a==b]
        #acc_top1_cal=sum(z)/batch_size
        # Update the class normalized accuracy
        labels = net.blobs['label'].data
        predicted_labels = np.argmax(net.blobs[nameProbsLayer].data, axis=1)

        for pred, lbl in zip(predicted_labels, labels):
            acc_norm_top1[int(lbl)].append(float(pred==lbl))
            y_true.append(lbl)
            y_pred.append(pred)
        #pdb.set_trace()

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

    return y_true, y_pred, confusion_matrix(y_true, y_pred)


# net.blobs['loss3/top-5'].data
# net.blobs['loss3/top-1'].data
# net.blobs['loss3/classifier_foodCAT_googlenet_food101'].data

##### USAGE EXAMPLE (from ipython)
# import caffeWrapper

# y_true, y_pred, confusion_matrix = 
# caffeWrapper.testAccuracy('foodCAT_googlenet_food101', 'net_TEST', 'loss3/classifier_foodCAT_googlenet_food101', 'loss3/top-1', 'loss3/top-5')
# caffeWrapper.testAccuracy('foodCAT_googlenet_food101', 'net_TEST_just_foodCAT', 'loss3/classifier_foodCAT_googlenet_food101', 'loss3/top-1', 'loss3/top-5')

# caffeWrapper.testAccuracy('foodCAT_VGG_ILSVRC_19_layers', 'net_TEST', 'fc8_foodCAT', 'accuracy@1', 'accuracy@5')
# caffeWrapper.testAccuracy('foodCAT_VGG_ILSVRC_19_layers', 'net_TEST_just_foodCAT', 'fc8_foodCAT', 'accuracy@1', 'accuracy@5')
def testAccuracy(modelName, modelType, nameProbsLayer, nameAccuracyTop1, nameAccuracyTop5):
    """
        To be as general as posible script, we need to fit some args before execute:
	modelName: Name of the model to use, in order to get the '.caffemodel' file (see definition of dict 'models' for more understanding)
	modelType: In order to get the model definition depending on which dataset we'll use (posible options so far, net_TEST or net_TEST_just_foodCAT, where both are dicts on this code)
	nameProbsLayer: The top name of the last innerProduct blob used in the model definition
	nameAccuracyTop1: The top name of the Accuracy blob used in the model definition
	nameAccuracyTop5: The top name of the Accuracy blob used in the model definition with the parameter 'top_k: 5'
    """

    net = deployModel(modelName, modelType)
    batch_size = net.blobs['data'].num
    nImages = numImages[modelType]
    numClasses_to_predict = numClass[modelType]
    num_batches = nImages/batch_size #TODO this should be calculated automatically from the net, not with our declaration 'nImages'
    print 'Total number of images to Test: ', nImages, ' in batches of: ', batch_size
    print 'Required Iterations: ', num_batches
    return check_accuracy(net, num_batches, batch_size, numClasses_to_predict, nameProbsLayer, nameAccuracyTop1, nameAccuracyTop5)

if __name__ == "__main__":

    net = deployModel(sys.argv[1])
