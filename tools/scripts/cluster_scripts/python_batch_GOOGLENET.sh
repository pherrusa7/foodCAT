#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=googlenet
#SBATCH --mem=8000
module add caffe/rc3-foss-2016a-CUDA-7.5.18
caffe train -solver models/googlenet_resized/solver_from_imagenet.prototxt -weights models/googlenet_resized/snapshots_from_imagenet/bvlc_googlenet.caffemodel
