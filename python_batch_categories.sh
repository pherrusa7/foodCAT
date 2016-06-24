#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=googlenet_categories
#SBATCH --mem=8000
module add caffe/rc3-foss-2016a-CUDA-7.5.18
caffe train -solver models/googlenet_categories/solver.prototxt -weights models/googlenet_categories/snapshots/foodRecognition_googlenet_finetunning_v2_1_iter_448000.caffemodel
