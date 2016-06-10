#!/bin/bash
#SBATCH --time=26:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=caffe_alexnet
#SBATCH --mem=8000
module add caffe/rc3-foss-2016a-CUDA-7.5.18
caffe train -solver models/googlenet_SR_balanced/solver.prototxt -weights models/googlenet_SR_balanced/snapshots/bvlc_googlenet.caffemodel
