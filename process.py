#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: process.py 
@time: 2019-09-11 23:29
@description:
"""
import os
import shutil
def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def process():
    path = ''
    train_filenames = os.listdir(path + 'cat_dog/train')
    train_cat = filter(lambda x: x[:3] == 'cat', train_filenames)
    train_dog = filter(lambda x: x[:3] == 'dog', train_filenames)

    valid_filenames = os.listdir(path + 'cat_dog/val')
    valid_cat = filter(lambda x: x[:3] == 'cat', valid_filenames)
    valid_dog = filter(lambda x: x[:3] == 'dog', valid_filenames)

    rmrf_mkdir(path + 'train2')
    os.mkdir(path + 'train2/cat')
    os.mkdir(path + 'train2/dog')

    rmrf_mkdir(path + 'valid2')
    os.mkdir(path + 'valid2/cat')
    os.mkdir(path + 'valid2/dog')

    # rmrf_mkdir(path + 'test2')
    # os.symlink('cat_dog/test/', 'test2/test')
    shutil.copytree(path + 'cat_dog/test/', path + 'test2/test')

    for filename in train_cat:
        # os.symlink('cat_dog/train/' + filename, 'train2/cat/' + filename)
        shutil.copy(path + 'cat_dog/train/' + filename, path + 'train2/cat/' + filename)

    for filename in train_dog:
        # os.symlink('cat_dog/train/' + filename, 'train2/dog/' + filename)
        shutil.copy(path + 'cat_dog/train/' + filename, path + 'train2/dog/' + filename)

    for filename in valid_cat:
        # os.symlink('cat_dog/val/' + filename, 'valid2/cat/' + filename)
        shutil.copy(path + 'cat_dog/val/' + filename, path + 'valid2/cat/' + filename)

    for filename in valid_dog:
        # os.symlink('cat_dog/val/' + filename, 'valid2/dog/' + filename)
        shutil.copy(path + 'cat_dog/val/' + filename, path + 'valid2/dog/' + filename)



if __name__ == '__main__':
    process()