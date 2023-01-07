import os
import re
import sys
import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
# from sklearn.metrics.cluster._supervised import roc_auc_score, accuracy_score
from PIL import Image
from dataset.load_dataset import *
from models.resnet_cbam import ResidualNet
from models.resnet_active import arlnet50
from models.densenet_cbam import densenet121_cbam
from models.densenet_active import densenet121_active
from utils import *
from models.mymodels import *
import argparse

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
parser.add_argument('-combine', default='RR', type=str, choices=['RD', 'DR', 'RR', 'DD'],help='combination type (RR)')
parser.add_argument('-extra', type=str, default='_avgpool', metavar='BS', help='(default: 123)')
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
    # transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(),
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
first_model = None
second_model = None
fusion_model = None
if args.combine == 'RD':
    first_model = arlnet50(pretrained=True)
    second_model = densenet121_cbam()
    fusion_model = model_fusion(2048)
elif args.combine == 'DR':
    first_model = densenet121_active(pretrained=True)
    second_model = ResidualNet(network_type="ImageNet", depth=50, num_classes=3, att_type='CBAM')
    fusion_model = model_fusion(2048)
elif args.combine == 'RR':  # *************************** Best ***********************************
    first_model = arlnet50(pretrained=True)
    second_model = ResidualNet(network_type="ImageNet", depth=50, num_classes=3, att_type='CBAM')
    fusion_model = model_fusion(4096)
elif args.combine == 'DD':
    first_model = densenet121_active(pretrained=True)
    second_model = densenet121_cbam()
    fusion_model = model_fusion(2048)

temp_model = DMAL(256)

# optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# # 1    7133 normal
# # 0    2531 covid
# # 2     942 viral_pneumonia
covid = 2531.
normal = 7133.
viral_pneumonia = 942.
weights = [normal/covid, normal/normal, normal/viral_pneumonia]
class_weights = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# criterion = nn.CrossEntropyLoss()
first_optimizer = optim.Adam(first_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
second_optimizer = optim.Adam(second_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
temp_optimizer = optim.Adam(temp_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
fusion_optimizer = optim.Adam(fusion_model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

first_model.to(device)
second_model.to(device)
temp_model.to(device)
fusion_model.to(device)
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

# ****************** first_train *******************
print()
# with open(logFile_train, "a") as f:
#     print("********* train_first ***********", file=f)
# with open(logFile_val, "a") as f:
#     print("********* val_first ***********", file=f)
# with open(logFile_test, "a") as f:
#     print("********* test_first ***********", file=f)
#
# best_val_acc = 0
# best_epoch = 0
# count = 0
# for epoch in range(args.num_epochs):
#
#     # train
#     print("********* train_first ***********")
#     train_first(loader_train, first_model, criterion, first_optimizer, device, logFile_train, epoch)
#
#     # evaluate
#     print("********* val_first ***********")
#     acc_val = evaluate_one(loader_val, first_model, criterion, device, logFile_val, epoch)
#
#     if acc_val > best_val_acc:
#         print("********* test_first ***********")
#         acc_test = evaluate_one(loader_test, first_model, criterion, device, logFile_test, epoch)
#         count = 0
#         best_val_acc = acc_val
#         best_epoch = epoch
#     else:
#         count += 1
#
#     if count == 5:
#         name_model = 'model_first.pt'
#         path = os.path.join('./models_params', args.combine, name_model)
#         torch.save(first_model.state_dict(), path)
#         print("Converged at epoch:%d with ACC:%.4f" % (best_epoch, best_val_acc))
#         break
with open(logFile_train, "a") as f:
    print("********* train_first ***********", file=f)
with open(logFile_val, "a") as f:
    print("********* val_first ***********", file=f)
with open(logFile_test, "a") as f:
    print("********* test_first ***********", file=f)

best_acc = 0
best_epoch = 0
for epoch in range(args.num_epochs):

    # train
    print("********* train_first ***********")
    train_first(loader_train, first_model, criterion, first_optimizer, device, logFile_train, epoch)

    print("********* test_first ***********")
    acc_test = evaluate_one(loader_test, first_model, criterion, device, logFile_test, epoch)

    if acc_test > best_acc:
        best_acc = acc_test
        best_epoch = epoch

    if epoch == args.num_epochs-1:
        p = os.path.join('./models_params', args.combine+args.extra)
        if not os.path.isdir(p):
            os.makedirs(p)
        name_model = 'model_first.pt'
        path = os.path.join(p, name_model)
        torch.save(first_model.state_dict(), path)
        print("Converged at epoch:%d with ACC:%.4f" % (best_epoch, best_acc))
        break




# ****************** second_train *******************
print()
# with open(logFile_train, "a") as f:
#     print("********* train_second ***********", file=f)
# with open(logFile_val, "a") as f:
#     print("********* val_second ***********", file=f)
# with open(logFile_test, "a") as f:
#     print("********* test_second ***********", file=f)
#
# best_val_acc = 0
# best_epoch = 0
# count = 0
# for epoch in range(args.num_epochs):
#
#     # train
#     print("********* train_second ***********")
#     train_second(loader_train, first_model, temp_model, second_model, criterion, temp_optimizer, second_optimizer,
#                  device, logFile_train, epoch)
#
#     # evaluate
#     print("********* val_second ***********")
#     acc_val = evaluate_two(loader_val, first_model, temp_model, second_model, criterion, device, logFile_val, epoch)
#
#     if acc_val > best_val_acc:
#         print("********* test_second ***********")
#         acc_test = evaluate_two(loader_test, first_model, temp_model, second_model, criterion, device, logFile_test,
#                                 epoch)
#         count = 0
#         best_val_acc = acc_val
#         best_epoch = epoch
#     else:
#         count += 1
#
#     if count == 5:
#         name_model = 'model_temp.pt'
#         name_model2 = 'model_second.pt'
#         path = os.path.join('./models_params', args.combine, name_model)
#         path2 = os.path.join('./models_params', args.combine, name_model2)
#         torch.save(temp_model.state_dict(), path)
#         torch.save(second_model.state_dict(), path2)
#         print("Converged at epoch:%d with ACC:%.4f" % (best_epoch, best_val_acc))
#         break
with open(logFile_train, "a") as f:
    print("********* train_second ***********", file=f)
with open(logFile_val, "a") as f:
    print("********* val_second ***********", file=f)
with open(logFile_test, "a") as f:
    print("********* test_second ***********", file=f)

best_acc = 0
best_epoch = 0
for epoch in range(args.num_epochs):

    # train
    print("********* train_second ***********")
    train_second(loader_train, first_model, temp_model, second_model, criterion, temp_optimizer, second_optimizer,
                 device, logFile_train, epoch)

    print("********* test_second ***********")
    acc_test = evaluate_two(loader_test, first_model, temp_model, second_model, criterion, device, logFile_test,
                            epoch)
    if acc_test > best_acc:
        best_acc = acc_test
        best_epoch = epoch

    if epoch == args.num_epochs-1:
        p = os.path.join('./models_params', args.combine+args.extra)
        if not os.path.isdir(p):
            os.makedirs(p)
        name_model = 'model_temp.pt'
        name_model2 = 'model_second.pt'
        path = os.path.join(p, name_model)
        path2 = os.path.join(p, name_model2)
        torch.save(temp_model.state_dict(), path)
        torch.save(second_model.state_dict(), path2)
        print("Converged at epoch:%d with ACC:%.4f" % (best_epoch, best_acc))
        break





# ****************** fusion_train *******************
with open(logFile_train, "a") as f:
    print("********* train_fusion ***********", file=f)
with open(logFile_val, "a") as f:
    print("********* val_fusion ***********", file=f)
with open(logFile_test, "a") as f:
    print("********* test_fusion ***********", file=f)

best_acc = 0
best_epoch = 0
for epoch in range(args.num_epochs):

    # train
    print("********* train_fusion ***********")
    train_fusion(loader_train, first_model, temp_model, second_model, fusion_model, criterion, fusion_optimizer,
                 device, logFile_train, epoch)

    print("********* test_fusion ***********")
    acc_test = evaluate_fusion(loader_test, first_model, temp_model, second_model, fusion_model, criterion, device,
                               logFile_test, epoch)

    if acc_test > best_acc:
        best_acc = acc_test
        best_epoch = epoch

    if epoch == args.num_epochs-1:
        p = os.path.join('./models_params', args.combine+args.extra)
        if not os.path.isdir(p):
            os.makedirs(p)
        name_model = 'model_fusion.pt'
        path = os.path.join(p, name_model)
        torch.save(fusion_model.state_dict(), path)
        print("Converged at epoch:%d with ACC:%.4f" % (best_epoch, best_acc))
        break




# # ****************** last_train *******************
# with open(logFile_train, "a") as f:
#     print("********* train_last ***********", file=f)
# with open(logFile_val, "a") as f:
#     print("********* val_last ***********", file=f)
# with open(logFile_test, "a") as f:
#     print("********* test_last ***********", file=f)
#
# # name_model = 'model_first.pt'
# # name_model2 = 'model_temp.pt'
# # name_model3 = 'model_second.pt'
# # name_model4 = 'model_fusion.pt'
# # ans = 'last'
# # path = os.path.join('./models_params', args.combine, name_model)
# # path2 = os.path.join('./models_params', args.combine, name_model2)
# # path3 = os.path.join('./models_params', args.combine, name_model3)
# # path4 = os.path.join('./models_params', args.combine, name_model4)
# # first_model.load_state_dict(torch.load(path))
# # temp_model.load_state_dict(torch.load(path2))
# # second_model.load_state_dict(torch.load(path3))
# # fusion_model.load_state_dict(torch.load(path4))
# # first_model = first_model.to(device)
# # temp_model = temp_model.to(device)
# # second_model = second_model.to(device)
# # fusion_model = fusion_model.to(device)
#
# best_val_acc = 0
# best_epoch = 0
# count = 0
# for epoch in range(args.num_epochs):
#
#     # train
#     print("********* train_last ***********")
#     train_last(loader_train, first_model, temp_model, second_model, fusion_model, criterion, first_optimizer,
#                temp_optimizer, second_optimizer, fusion_optimizer, device, logFile_train, epoch)
#
#     # evaluate
#     print("********* val_last ***********")
#     acc_val = evaluate_fusion(loader_val, first_model, temp_model, second_model, fusion_model, criterion, device,
#                               logFile_val, epoch)
#
#     if acc_val > best_val_acc:
#         print("********* test_last ***********")
#         acc_test = evaluate_fusion(loader_test, first_model, temp_model, second_model, fusion_model, criterion, device,
#                                    logFile_test, epoch)
#         count = 0
#         best_val_acc = acc_val
#         best_epoch = epoch
#     else:
#         count += 1
#
#     if count == 5:
#         name_model = 'last_model_first.pt'
#         name_model2 = 'last_model_temp.pt'
#         name_model3 = 'last_model_second.pt'
#         name_model4 = 'last_model_fusion.pt'
#         ans = 'last'
#         path = os.path.join('./models_params', args.combine, ans, name_model)
#         path2 = os.path.join('./models_params', args.combine, ans, name_model2)
#         path3 = os.path.join('./models_params', args.combine, ans, name_model3)
#         path4 = os.path.join('./models_params', args.combine, ans, name_model4)
#         torch.save(first_model.state_dict(), path)
#         torch.save(temp_model.state_dict(), path2)
#         torch.save(second_model.state_dict(), path3)
#         torch.save(fusion_model.state_dict(), path4)
#         print("Converged at epoch:%d with ACC:%.4f" % (best_epoch, best_val_acc))
#         break
