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
####################### END TEST ZONE




####################### DEPLOY, to use it as the other people. REQUIRE IMAGE PREPROCESSING AS CAFFE DOES IN TRAIN. (OR NOT???)
deploy  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
	   "foodCAT_googlenet_food101":	  os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/deploy.prototxt"),
	   "foodCAT_VGG_ILSVRC_19_layers": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/CLUSTER/VGG_ILSVRC_19_layers_deploy.prototxt")}

mean  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
	   "foodCAT_googlenet_food101":	  [104, 117, 123],
	   "foodCAT_VGG_ILSVRC_19_layers": [104, 117, 123]}
####################### END DEPLOY


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

def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size, nameProbsLayer, nameAccuracyTop1, nameAccuracyTop5):
    ### Set variables
    # We get the number of classes by the size of the vector of the first prediction
    numClasses = len(net.blobs[nameProbsLayer].data[0])
    # Inicialize the accuracy reported by the net
    acc_top1 = 0
    acc_top5 = 0
    # Inicialize the array to save the accuracy of all our examples
    acc_norm_top1 = [[] for i in xrange(numClasses)]

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

        #pdb.set_trace()

    acc_top1 = acc_top1/num_batches
    acc_top5 = acc_top5/num_batches
    print 'acc_top1: ', acc_top1
    print 'acc_top5: ', acc_top5
    print 'acc_norm_top1: ', sum( [sum(classAccur)/len(classAccur) for classAccur in acc_norm_top1 if len(classAccur)>0] )/numClasses

    print '#images tested: ', sum([len(classAccur) for classAccur in acc_norm_top1])
    print '#images supose to test: ', num_batches*batch_size


# net.blobs['loss3/top-5'].data
# net.blobs['loss3/top-1'].data
# net.blobs['loss3/classifier_foodCAT_googlenet_food101'].data


# caffeWrapper.testAccuracy('foodCAT_googlenet_food101', 'net_TEST', 'loss3/classifier_foodCAT_googlenet_food101', 'loss3/top-1', 'loss3/top-5')
# caffeWrapper.testAccuracy('foodCAT_googlenet_food101', 'net_TEST_just_foodCAT', 'loss3/classifier_foodCAT_googlenet_food101', 'loss3/top-1', 'loss3/top-5')

# caffeWrapper.testAccuracy('foodCAT_VGG_ILSVRC_19_layers', 'net_TEST', 'fc8_foodCAT', 'accuracy@1', 'accuracy@5')
# caffeWrapper.testAccuracy('foodCAT_VGG_ILSVRC_19_layers', 'net_TEST_just_foodCAT', 'fc8_foodCAT', 'accuracy@1', 'accuracy@5')




def testAccuracy(modelName, modelType, nameProbsLayer, nameAccuracyTop1, nameAccuracyTop5):
    net = deployModel(modelName, modelType)
    batch_size = net.blobs['data'].num
    nImages = numImages[modelType]
    print 'Total number of images to Test: ', nImages, ' in batches of: ', batch_size
    print 'Required Iterations: ', nImages/batch_size  #TODO this should be calculated automatically from the net

    check_accuracy(net, nImages/batch_size, batch_size, nameProbsLayer, nameAccuracyTop1, nameAccuracyTop5)

if __name__ == "__main__":

    net = deployModel(sys.argv[1])
