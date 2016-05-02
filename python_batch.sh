#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=caffe_VGG
#SBATCH --mem=4000
module add caffe/rc3-foss-2016a-CUDA-7.5.18
caffe train -solver models/foodCAT_VGG_ILSVRC_19_layers/CLUSTER/solver.prototxt --snapshot=models/foodCAT_VGG_ILSVRC_19_layers/snapshots/ss_foodCAT_VGG_ILSVRC_19_layers_train_iter_50000.solverstate
