#!/usr/bin/env python
# coding: utf-8
#ResNet for Classify leaves

# import packages
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib
import matplotlib.pyplot as plt
import collections
import math
import os
import shutil
import pandas as pd
import os ; os. environ ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# define "Residual" class
class Residual(nn.Module): #@save
    def __init__(self, input_channels, num_channels, use_1x1conv = False, strides = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size = 3, padding = 1, stride = strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size = 3, padding = 1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size = 1, stride = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)    

# check the size of input and output
# double the channels, the size of output will be halved
X = torch.rand(4,3,6,6)
blk1 = Residual(3,3)
blk2 = Residual(3,6,use_1x1conv = True, strides = 2)
blk1(X).shape, blk2(X).shape

# Build ResNet Model

#b1 block
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
#resnet_block
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
    else:
        blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3 = nn.Sequential(*resnet_block(64,128,2))
b4 = nn.Sequential(*resnet_block(128,256,2))
b5 = nn.Sequential(*resnet_block(256,512,2))

net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(), nn.Linear(512,10))

# check the change of the size in different modules 
#X = torch.rand(1,1,224,224)
#for layer in net:
#    X = layer(X)
#    print(layer.__class__.__name__,'output shape:\t', X.shape)

# Train the model
#lr, num_epochs, batch_size = 0.05, 10, 64
#train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
#d2l.train_ch6(net,train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

def get_net():
    num_classes = len(set(labels.values()))
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")


# Re-organize The Dataset

# total images: 27153
#train 0-18352.jpg  \\  test 18353-27152.jpg (8800)

#data_dir = '../classify-leaves'
data_dir = '/scratch/c.c21021656/Datasets/ClaLeaves2/classify-leaves'

#read the labels
def read_csv_labels(fname):
    """read `fname` to return a label"""
    with open(fname, 'r') as f:
        # skip the first line
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir,'train.csv'))

print('# train examples :', len(labels))
print('# categories :', len(set(labels.values())))
print(data_dir)



def reorg_leaf_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'train.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)
    
batch_size = 64
valid_ratio = 0.1 # 10% of train images are used for validation
#reorg_leaf_data(data_dir, valid_ratio)



#data augmentation
transform_train = torchvision.transforms.Compose([
    # randomly croped the image
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),ratio=(3.0 / 4.0, 4.0 / 3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # randomly change the brightness,contrast,saturation
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
    # 
    torchvision.transforms.ToTensor(),
    # normalization
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # crop from the centre of image
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

print('Finished data augmentation')
# In[11]:


#Read the dataset
train_ds, train_valid_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]

print('Finished read the dataset')
# In[12]:


# create data loader
train_iter, train_valid_iter = [
    torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,drop_last=False)



# Train Function
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],legend=['train loss', 'train acc', 'valid acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels, loss,trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,(metric[0] / metric[2], metric[1] / metric[2], None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    if valid_iter is not None:
        print(f'loss {metric[0] / metric[2]:.3f}, '
              f'train acc {metric[1] / metric[2]:.3f}, '
              f'valid acc {valid_acc:.3f}')
    else:
        print(f'loss {metric[0] / metric[2]:.3f}, '
              f'train acc {metric[1] / metric[2]:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')


# In[ ]:
print('Start training, hope it is fast :)')

# Train and varify the model !!
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 80, 0.1, 5e-4
lr_period, lr_decay,net = 50, 0.1,get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,lr_decay)

print('Finished, all done!!!')