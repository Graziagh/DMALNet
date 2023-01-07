import os
import re
import sys
import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image
from dataset.load_dataset import *
from models.resnet_cbam import ResidualNet
from models.resnet_active import arlnet50
from models.densenet_cbam import densenet121_cbam
from models.densenet_active import densenet121_active
from utils import *
from models.mymodels import *
import argparse
from models.model_other import choose_model

parser = argparse.ArgumentParser(description='COVID-19 Image Training')

parser.add_argument('-depth', default=50, type=int, metavar='D', help='model depth')
parser.add_argument('-ngpu', default=4, type=int, metavar='G', help='number of gpus to use')
parser.add_argument('-workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-num_epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-model_type', default='efficientnet', type=str, choices=
['alexnet', 'vgg', 'squeezenet', 'densenet', 'mobilenet', 'inception', 'googlenet', 'xception', 'resnet', 'efficientnet'],help='combination type (RD)')
parser.add_argument("-seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument('-combine', default='R', type=str, choices=['RD', 'DR', 'RR', 'DD'],help='combination type (RD)')
parser.add_argument('-extra', type=str, default='_S', metavar='BS', help='(default: 123)')
# parser.add_argument("-combin", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
# parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
# parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)
args = parser.parse_args()

print('batch_size:', args.batch_size)
print('num_epochs:', args.num_epochs)
print('combine:', args.combine)
print('workers:', args.workers)


#image net mean values
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

image_transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    #transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

# dataset
dataset_train = COVID19(data_dir='./dataset/data/', filename='train', transform=image_transformations)
dataset_val = COVID19(data_dir='./dataset/data/', filename='val', transform=image_transformations)
dataset_test = COVID19(data_dir='./dataset/data/', filename='test', transform=image_transformations)


# dataloader
loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

# model component
# model = arlnet50(pretrained=True)
model = ResidualNet(network_type="ImageNet", depth=50, num_classes=3, att_type='CBAM')

# optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


covid = 2531.
normal = 7133.
viral_pneumonia = 942.
weights = [normal/covid, normal/normal, normal/viral_pneumonia]
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

model.to(device)
criterion.to(device)


# train process
# logFile_train = args.combine + '_train_' + time.strftime("%Y%m%d_%H_%M") + '.txt'
p = os.path.join('./text', args.combine+args.extra)
if not os.path.isdir(p):
    os.makedirs(p)
print(p)
logFile_train = os.path.join(p, 'train.txt')
logFile_val = os.path.join(p, 'val.txt')
logFile_test = os.path.join(p, 'test.txt')
# makeLogFile(logFile_train)
# makeLogFile(logFile_val)
# makeLogFile(logFile_test)

# ****************** train_other *******************
with open(logFile_train, "a") as f:
    print("********* train_other ***********", file=f)
with open(logFile_val, "a") as f:
    print("********* val_other ***********", file=f)
with open(logFile_test, "a") as f:
    print("********* test_other ***********", file=f)

best_acc = 0
best_epoch = 0
for epoch in range(args.num_epochs):

    # train
    print("********* train ***********")
    train_other_model(loader_train, model, criterion, optimizer, device, logFile_train, epoch)

    print("********* test ***********")
    acc_test = evaluate_other_model(loader_test, model, criterion, device, logFile_test, epoch)

    if acc_test > best_acc:
        best_acc = acc_test
        best_epoch = epoch

    if epoch == args.num_epochs-1:
        p = os.path.join('./models_params', args.combine + args.extra)
        if not os.path.isdir(p):
            os.makedirs(p)
        name_model = 'model.pt'
        path = os.path.join(p, name_model)
        torch.save(model.state_dict(), path)
        print("Converged at epoch:%d with ACC:%.4f" % (best_epoch, best_acc))
        break



