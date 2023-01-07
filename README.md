# DMALNet
An improved COVID-19 classification model on chest radiography by dual-ended multiple attention learning

# Introduction
train.py shows the parameters of our model and the framework for training

utils.py contains a specific process for step-by-step training and assessment as well as various tool functions

# Environment
Pytorch 3.8

# Run
python train.py -num_epochs 100 -batch_size 512 -workers 4

# Data
COVID-19 Radiography Database: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

# Acknowledgements
+ M. E. H. Chowdhury et al., _Can AI Help in Screening Viral and COVID-19 Pneumonia_, in IEEE Access, vol. 8, pp. 132665-132676, 2020, doi: 10.1109/ACCESS.2020.3010287.
+ Rahman T, Khandakar A, Qiblawey Y et al., _Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images_, Computers in biology and medicine, 2021, 132: 104319.

