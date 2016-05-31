#!/bin/bash
#SBATCH --time=26:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=caffe_alexnet
#SBATCH --mem=8000
module add caffe/rc3-foss-2016a-CUDA-7.5.18
caffe train -solver models/googlenet_resized/solver.prototxt -weights models/googlenet_resized/snapshots/foodRecognition_googlenet_finetunning_v2_1_iter_448000.caffemodel
