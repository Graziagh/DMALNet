import torch
from models.mymodels import *
from models.model_other import *
import os
# from models.densenet_cbam import densenet121_cbam

# model = ResidualNet(network_type="ImageNet", depth=50, num_classes=3, att_type='CBAM')
# model = arlnet50(pretrained=True)
# model = densenet121_active(pretrained=True)
# model = densenet121_cbam()
# model=densenet121(pretrained=False)
# model = choose_model('efficientnet')
# # for i,m in enumerate(model.modules()):
# #     print(i,m)
# print(model)

# for x in range(1,4):
#     print(x)

# m = nn.Softmax2d()
# # 你在 2 维上进行 softmax
# input = torch.randn([2,2,3,4])
# print(input)
# output = m(input)
# print(output)

# model = DMAL(50)
# # print(model)
# x1 = torch.randn((10,10,5,5))
# x2 = torch.randn((10,10,5,5))
# x3 = torch.randn((10,10,5,5))
# out = build_img(model,x1,x2,x3)
# print(out.shape)

# model = model_fusion()
# print(model)
# import torch
# print(torch.__version__)
#
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
#
# print(torch.cuda.is_available())
# #cuda是否可用；
#
# print(torch.cuda.device_count())
# #返回gpu数量；
#
# print(torch.cuda.get_device_name(0))
# #返回gpu名字，设备索引默认从0开始；
#
# print(torch.cuda.current_device())
#
#
# a = torch.cuda.is_available()
# print(a)

# model = DMAL(50)
# a = torch.randn((2,8,8,8))
# b = torch.randn((2,4,4,4))
# c = torch.randn((2,2,2,2))
# build_img(model,a,b,c)

# p = os.path.join('./text', '11')
# a = os.path.isdir(p)
# print(a)

# combine = 'RD'
# p = os.path.join('./models_params', combine + '134')
# if not os.path.isdir(p):
#     os.makedirs(p)
# name_model = 'model_fusion.pt'
# path = os.path.join(p, name_model)
# print(path)

# import sklearn
# print(sklearn.__version__)

import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
# logsoft-max + NLLLoss
m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(m(input), target)
print('logsoftmax + nllloss output is {}'.format(output))

# crossentripyloss
loss = nn.CrossEntropyLoss()
# input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(input, target)
print('crossentropy output is {}'.format(output))


# one hot label loss
C = 5
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
print('target is {}'.format(target))
N = target .size(0)
# N 是batch-size大小
# C is the number of classes.
labels = torch.full(size=(N, C), fill_value=0)
print('labels shape is {}'.format(labels.shape))
labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1)
print('labels is {}'.format(labels))

log_prob = m(input)
loss = -torch.sum(log_prob * labels) / N
print('N is {}'.format(N))
print('one-hot loss is {}'.format(loss))



# one hot label loss
C = 3
N = labels.size(0)
# N 是batch-size大小
# C is the number of classes.
labels2 = torch.full(size=(N, C), fill_value=0)
labels2.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1)
m = nn.LogSoftmax(dim=1)
log_prob = m(input)
loss = -torch.sum(log_prob * labels2) / N
