#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import argparse, math, time, json, os

from lib import wrn, transform
from config import config
import vis


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="cifar10", type=str, help="dataset name : [svhn, cifar10]")
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--poison-data-ratio", default=0.0, type=float)




args = parser.parse_args()



plotter = vis.Plotter()

print("dataset : {}".format(args.dataset))


dataset_cfg = config[args.dataset]

l_train_dataset = dataset_cfg["dataset"](args.root, "l_train")
u_train_dataset = dataset_cfg["dataset"](args.root, "u_train")


images = l_train_dataset.dataset['images']
labels = l_train_dataset.dataset['labels']

print(images.shape, images.dtype, np.max(images), np.min(images))
print(labels.shape, labels.dtype, np.unique(labels), np.max(labels), np.min(labels))

for c in np.unique(labels):
    print("\t{} : {}".format(c, np.sum(labels == c)))

images = u_train_dataset.dataset['images']
labels = u_train_dataset.dataset['labels']

print(images.shape, images.dtype, np.max(images), np.min(images))
print(labels.shape, labels.dtype, np.unique(labels), np.max(labels), np.min(labels))

for c in np.unique(labels):
    print("\t{} : {}".format(c, np.sum(labels == c)))



