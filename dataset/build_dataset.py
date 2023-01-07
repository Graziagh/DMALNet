# libraries for files preparation
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import shutil

# libraries for CNN models and plotting
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import torchvision.transforms as transforms
import os
import cv2 as cv
import torch

import pandas as pd
import numpy as np
from PIL import Image

# load data
covid = pd.read_excel(r"E:\dataset\archive\COVID-19_Radiography_Dataset\COVID.metadata.xlsx")
# covid.head()
normal = pd.read_excel(r"E:\dataset\archive\COVID-19_Radiography_Dataset\Normal.metadata.xlsx")
# normal.head()
viral_pneumonia = pd.read_excel(r"E:\dataset\archive\COVID-19_Radiography_Dataset\Viral Pneumonia.metadata.xlsx")
viral_pneumonia.head()
# Lung_Opacity.head()


# print("Covid cases: ", str(len(covid)))
# print("Normal cases: ", str(len(normal)))
# print("Viral Pneumonia cases: ", str(len(viral_pneumonia)))

# add label for each case
covid['label'] = 0
normal['label'] = 1
viral_pneumonia['label'] = 2

# drop non-related columns
covid = covid[['FILE NAME', 'label']]
normal = normal[['FILE NAME', 'label']]
viral_pneumonia = viral_pneumonia[['FILE NAME', 'label']]

# print(covid.head())

# sampling data for covid and normal cases
# df_0 = covid.sample(SAMPLE_SIZE, random_state=26)
# df_1 = normal.sample(SAMPLE_SIZE, random_state=26)

# concat dataframes
data = pd.concat([covid, normal, viral_pneumonia], axis=0).reset_index(drop=True)

# check numbers of each label
# print (data['label'].value_counts())

# shuffle data
data = shuffle(data)
# data.head()

# data divide
df_train1, df_test = train_test_split(data, test_size=0.20, random_state=26, stratify=data['label'])
df_train, df_val = train_test_split(df_train1, test_size=1/8, random_state=26, stratify=df_train1['label'])

# datasample divide
# df_data, df_remain = train_test_split(data, test_size=0.995, random_state=26, stratify=data['label'])
# df_train1, df_test = train_test_split(df_data, test_size=0.20, random_state=26, stratify=df_data['label'])
# df_train, df_val = train_test_split(df_train1, test_size=1/8, random_state=26, stratify=df_train1['label'])

# print(df_train.shape)
# print(df_val.shape)
# print(df_test.shape)
# print(df_train.head())

train_list = list(df_train['FILE NAME'])
val_list = list(df_val['FILE NAME'])
test_list = list(df_test['FILE NAME'])

train_image_path = []
train_label = []
for image in train_list:
    fname = image + '.png'
    lab = int(df_train.loc[df_train['FILE NAME'] == image, ['label']].values)
    if lab == 0:
        src = os.path.join(r"E:\dataset\archive\COVID-19_Radiography_Dataset\COVID\images", fname)
        # label = [1, 0, 0]
    elif lab == 1:
        fname = fname.capitalize()
        src = os.path.join(r"E:\dataset\archive\COVID-19_Radiography_Dataset\Normal\images", fname)
        # label = [0, 1, 0]
    elif lab == 2:
        src = os.path.join(r"E:\dataset\archive\COVID-19_Radiography_Dataset\Viral Pneumonia\images", fname)
        # label = [0, 0, 1]

    train_image_path.append(src)
    train_label.append(lab)

print('train done...')

val_image_path = []
val_label = []
for image in val_list:
    fname = image + '.png'
    lab = int(df_val.loc[df_val['FILE NAME'] == image, ['label']].values)
    if lab == 0:
        src = os.path.join(r"E:\dataset\archive\COVID-19_Radiography_Dataset\COVID\images", fname)
        # label = [1, 0, 0]
    elif lab == 1:
        fname = fname.capitalize()
        src = os.path.join(r"E:\dataset\archive\COVID-19_Radiography_Dataset\Normal\images", fname)
        # label = [0, 1, 0]
    elif lab == 2:
        src = os.path.join(r"E:\dataset\archive\COVID-19_Radiography_Dataset\Viral Pneumonia\images", fname)
        # label = [0, 0, 1]

    val_image_path.append(src)
    val_label.append(lab)

print('val done...')

test_image_path = []
test_label = []
for image in test_list:
    fname = image + '.png'
    lab = int(df_test.loc[df_test['FILE NAME'] == image, ['label']].values)
    if lab == 0:
        src = os.path.join(r"E:\dataset\archive\COVID-19_Radiography_Dataset\COVID\images", fname)
        # label = [1, 0, 0]
    elif lab == 1:
        fname = fname.capitalize()
        src = os.path.join(r"E:\dataset\archive\COVID-19_Radiography_Dataset\Normal\images", fname)
        # label = [0, 1, 0]
    elif lab == 2:
        src = os.path.join(r"E:\dataset\archive\COVID-19_Radiography_Dataset\Viral Pneumonia\images", fname)
        # label = [0, 0, 1]

    test_image_path.append(src)
    test_label.append(lab)

print('test done...')

# train_arr = [cv.imread(x) for x in train_image_path]
# train_tensor_list = [torch.from_numpy(x) for x in train_arr]
# train_tensor = torch.stack(train_tensor_list, dim=0)
# train_img_tensor = train_tensor.permute(0,3,1,2)
# print('train_img_tensor done...')
# train_label_arr = np.array(train_label)
# train_label_tensor = torch.from_numpy(train_label_arr)
# train_label_tensor = train_label_tensor.unsqueeze(dim=1)
# print('train_label_tensor done...')
# train = [train_img_tensor, train_label_tensor]
# torch.save(train, './data/train.pt')
# print('train.pt done...')
train_arr = [cv.imread(x) for x in train_image_path]
train_np = np.stack(train_arr, axis=0)
# train_img_np = train_np.transpose(0, 3, 1, 2)
print('train_img_np done...')
train_label_arr = np.array(train_label)
# train_label_np = np.expand_dims(train_label_arr, axis=1)
print('train_label_np done...')
np.savez('./data/train.npz', train=train_np, label=train_label_arr)
print('train.npz done...')



# valid_arr = [cv.imread(x) for x in val_image_path]
# valid_tensor_list = [torch.from_numpy(x) for x in valid_arr]
# valid_tensor = torch.stack(valid_tensor_list,dim=0)
# valid_img_tensor = valid_tensor.permute(0,3,1,2)
# print('valid_img_tensor done...')
# val_label_arr = np.array(val_label)
# val_label_tensor = torch.from_numpy(val_label_arr)
# val_label_tensor = val_label_tensor.unsqueeze(dim=1)
# print('val_label_tensor done...')
# valid = [valid_img_tensor, val_label_tensor]
# torch.save(valid, './data/val.pt')
# print('val.pt done...')
val_arr = [cv.imread(x) for x in val_image_path]
val_np = np.stack(val_arr, axis=0)
# val_img_np = val_np.transpose(0, 3, 1, 2)
print('val_img_np done...')
val_label_arr = np.array(val_label)
# val_label_np = np.expand_dims(val_label_arr, axis=1)
print('val_label_np done...')
np.savez('./data/val.npz', train=val_np, label=val_label_arr)
print('val.npz done...')



# test_arr = [cv.imread(x) for x in test_image_path]
# test_tensor_list = [torch.from_numpy(x) for x in test_arr]
# test_tensor = torch.stack(test_tensor_list,dim=0)
# test_img_tensor = test_tensor.permute(0,3,1,2)
# print('test_img_tensor done...')
# test_label_arr = np.array(test_label)
# test_label_tensor = torch.from_numpy(test_label_arr)
# test_label_tensor = test_label_tensor.unsqueeze(dim=1)
# print('test_label_tensor done...')
# test = [test_img_tensor, test_label_tensor]
# torch.save(test, './data/test.pt')
# print('test.pt done...')
test_arr = [cv.imread(x) for x in test_image_path]
test_np = np.stack(test_arr, axis=0)
# test_img_np = test_np.transpose(0, 3, 1, 2)
print('test_img_np done...')
test_label_arr = np.array(test_label)
# test_label_np = np.expand_dims(test_label_arr, axis=1)
print('test_label_np done...')
np.savez('./data/test.npz', train=test_np, label=test_label_arr)
print('test.npz done...')