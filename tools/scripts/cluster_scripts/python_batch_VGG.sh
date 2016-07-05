#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=caffe_alexnet
#SBATCH --mem=4000
module add caffe/rc3-foss-2016a-CUDA-7.5.18
caffe train -solver models/foodCAT_VGG_ILSVRC_19_layers_500/solver.prototxt -weights models/foodCAT_VGG_ILSVRC_19_layers_500/snapshots/ss_foodCAT_VGG_ILSVRC_19_layers_train_iter_80000.caffemodel

