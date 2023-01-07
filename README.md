# DMALNet
An improved COVID-19 classification model on chest radiography by dual-ended multiple attention learning

# Introduction
train.py shows the parameters of our model and the framework for training
utils.py contains a specific process for step-by-step training and assessment as well as various tool functions

# Environment
Pytorch 3.8

# Run
python train.py --num_epochs 100 --batch_size 512 --data_path XXX 


XXX denotes the path of the data set

# Data
COVID-19 Radiography Database: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

# Acknowledgements
+ Raghavendra Selvan, Erik B Dam, _Tensor Networks for Medical Image Classification_

