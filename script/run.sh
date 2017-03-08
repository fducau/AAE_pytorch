#!/bin/bash
source activate deep
python mnist_pytorch.py --elastic-augment True --momentum 0.4 --epochs 5 > output