"""
usage from TFG folder:  python tools/scripts/caffeWrapper.py

python tools/scripts/caffeWrapper.py foodCAT_googlenet_food101
python tools/scripts/caffeWrapper.py foodCAT_alexnet
python tools/scripts/caffeWrapper.py foodCAT_VGG_ILSVRC_19_layers
"""
import os
import sys
import caffe


PATH_TO_PROJECT='../..'
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
####################### END TEST ZONE




####################### DEPLOY, to use it as the other people. REQUIRE IMAGE PREPROCESSING AS CAFFE DOES IN TRAIN. (OR NOT???)

deploy  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
	   "foodCAT_googlenet_food101":	  os.path.join(PATH_TO_PROJECT, "models/foodCAT_googlenet_food101/deploy.prototxt"),
	   "foodCAT_VGG_ILSVRC_19_layers": os.path.join(PATH_TO_PROJECT, "models/foodCAT_VGG_ILSVRC_19_layers/CLUSTER/VGG_ILSVRC_19_layers_deploy.prototxt")}

mean  = {"foodCAT_alexnet":   os.path.join(PATH_TO_PROJECT, "models/foodCAT_alexnet/ TODO "),
	   "foodCAT_googlenet_food101":	  [104, 117, 123],
	   "foodCAT_VGG_ILSVRC_19_layers": [104, 117, 123]}

####################### END DEPLOY


def deployModel( model ):
    print models[model]

    # Set Caffe to GPU
    caffe.set_device(0)
    caffe.set_mode_gpu()
    #caffe.set_mode_cpu()

    model_def = deploy[model]
    model_weights = models[model]

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    return net


if __name__ == "__main__":

    deployModel(sys.argv[1])
