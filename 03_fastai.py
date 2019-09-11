#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 03_fastai.py 
@time: 2019-09-11 22:42
@description:
"""
from fastai import *
from fastai.vision import *
from fastai.vision.models import *
import numpy as np
import ssl

arch = resnet50


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


# path = untar_data(URLs.MNIST_SAMPLE)
# print(path)
#


def train_resnet18():
    path = 'data/'
    data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224)

    data.show_batch(3, figsize=(8, 8))
    plt.show()

    learn = cnn_learner(data, models.resnet152, metrics=accuracy)
    learn.fit(1)
    learn.export(file='resnet152.pkl')


def predict():
    learn = load_learner(path='data',file='resnet152.pkl',
                         test=ImageList.from_folder('test2/test'))
    preds, y = learn.get_preds(ds_type=DatasetType.Test)
    print(preds.data.numpy())
    preds=preds.data.numpy()
    y=np.argmax(preds,axis=1)
    print(y)

    sub = pd.DataFrame()
    filenames = os.listdir('test2/test')
    print(filenames)

    ids = [int(f.split('.')[0]) for f in filenames]
    sub['id'] = ids
    sub['label'] = y
    sub[['id', 'label']].to_csv('fastai.csv', index=None, header=False)

if __name__ == '__main__':
    train_resnet18()
    predict()
