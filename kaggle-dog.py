#!/usr/bin/env python
# coding: utf-8
# Kaggle£ºDog Breeed identification £¨ImageNet Dogs£©
# 120 categories, different size with CIFAR-10

# 10222 train images; 10357 test images.

# import packages and modules
import collections
#import d2lzh as d2l
from d2l import torch as d2l
import math
import torch
import torchvision
from mxnet import autograd, gluon, init, nd
from torch import nn
#from mxnet.gluon import data as gdata, loss as gloss, model_zoo, nn
from mxnet.gluon import model_zoo
import os
import shutil
import time
import zipfile


# 1.Unzip dataset.
print('-----###Part-1: Unzip datasets###-----Start')
# The dataset will not be unzipped or reorganized again if the flages below is set TRUE
DataIsUnzipped = True
DataIsReorg = True
data_dir = '/scratch/c.c21021656/Datasets/DogBreed'
if DataIsUnzipped:
    print('---Datas have been unzipped already.')
else:
    print('---Datas have NOT been unzipped, so will be done.')
    zipfiles = ['train.zip', 'test.zip', 'labels.csv.zip']
    #unzip train and test dataset
    for f in zipfiles[:-1]:
        with zipfile.ZipFile(data_dir + '/' + f, 'r') as z:
          z.extractall(data_dir + '/' + f.strip(".zip"))
    #unzip label csv
    f = zipfiles[len(zipfiles)-1]
    with zipfile.ZipFile(data_dir + '/' + f, 'r') as z:
      z.extractall(data_dir)
    print('---Finished unzip dataset, please check') 
print('-----###Part-1: Unzip datasets###-----End')


# 2.Reorganize dataset
print('-----###Part-2: Reorganize datasets###-----Start')
# `reorg_dog_data` is used to read train labels, cut validate set and reorganize test set
# 'valid_ratio' means the ratio between the validate number and the minimun label number in train dataset.
def reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
    #read the label csv
    with open(os.path.join(data_dir, label_file), 'r') as f:
        # skip the headline
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))
      
    # the minimum number of dog breed label in train dateset 
    min_n_train_per_label = (collections.Counter(idx_label.values()).most_common()[:-2:-1][0][1])
    print('min_n_train_per_label is ',min_n_train_per_label) 
    # label numbers of validate set
    n_valid_per_label = math.floor(min_n_train_per_label * valid_ratio)
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = train_file.split('.')[0]
        label = idx_label[idx]
        d2l.mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
        os.path.join(data_dir, input_dir, 'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            d2l.mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            d2l.mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
    # reorganize the test set
    d2l.mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test', 'unknown'))
#read the labels
def read_csv_labels(fname):
    """read `fname` to return a label"""
    with open(fname, 'r') as f:
        # skip the first line
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir,'labels.csv'))
print('# train examples :', len(labels))
print('# categories :', len(set(labels.values())))
print(data_dir)

label_file, train_dir, test_dir, input_dir = 'labels.csv', 'train', 'test','train_valid_test'

batch_size = 64

if DataIsReorg:
    print('---Datas have been reorganized already.')
else:
    valid_ratio = 0.1 
    print('---Datas have NOT been reorganized, so will be done.')
    reorg_dog_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)
    print('---Finished reorganized dataset, please check')
print('-----###Part-2: Reorganize datasets###-----End')

#3. Data augmentation

transform_train = torchvision.transforms.Compose([
    # randomly croped the image
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # randomly change the brightness,contrast,saturation
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # randomly add noise
    torchvision.transforms.ToTensor(),
    # normalize 3 channels
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# augment the test set
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # crop from the centre of image
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


#4. Load the dataset
print('-----###Part-3: Load datasets###-----Start')
# creat a instance `ImageFolderDataset` to load the image folder, and appled data augmentation.
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),
                                            transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),
                                            transform=transform_test) for folder in ['valid', 'test']]

# create a instance to load the dataset    
train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
                                for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)    
           
print('-----###Part-3: Load datasets###-----End')


#5. Define the model


def get_net(ctx):
    finetune_net = model_zoo.vision.resnet34_v2(pretrained=True)
    # define a new network
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # 120 is the cateories number
    finetune_net.output_new.add(nn.Dense(120))
    # initialize
    finetune_net.output_new.initialize(init.Xavier(), ctx=ctx)
    # sent parameters to graphics card
    finetune_net.collect_params().reset_ctx(ctx)
    return finetune_net

def get_net():
    num_classes = len(set(labels.values()))
    net = d2l.resnet18(num_classes, in_channels = 3)
    return net

#define loss function
loss = torch.nn.CrossEntropyLoss(reduction="none")
def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum = l.sum()
        n += labels.numel()
    return l_sum / n
    
    
# Train Function

def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels, loss,trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            print(f'loss {metric[0] / metric[2]:.3f}, 'f'train acc {metric[1] / metric[2]:.3f}, ')
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            print(f'valid acc {valid_acc:.3f}')
        scheduler.step()
    if valid_iter is not None:
        print(f'loss {metric[0] / metric[2]:.3f}, '
              f'train acc {metric[1] / metric[2]:.3f}, '
              f'valid acc {valid_acc:.3f}')
    else:
        print('Final result is : ' f'loss {metric[0] / metric[2]:.3f}, 'f'train acc {metric[1] / metric[2]:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(devices)}')



# train the model
print('-----###Part-4: Training###-----Start')
print('---try_all_gpus:',d2l.try_all_gpus())
ctx, num_epochs, lr, wd = d2l.try_all_gpus(), 80, 0.01, 1e-4
lr_period, lr_decay, net = 10, 0.1, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, ctx, lr_period, lr_decay)


# Classify the test set and generate output.
net = get_net()
train(net, train_valid_iter, None, num_epochs, lr, wd, ctx, lr_period,lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_context(ctx))
    output = nd.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(os.path.join(data_dir, input_dir, 'test/unknown')))
with open('submission2.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
            
print('-----###Part-4: Training###-----End')
