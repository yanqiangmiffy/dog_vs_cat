# https://github.com/ypwhs/dogs_vs_cats
# 未实现
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import h5py
import os
import shutil

train_filenames = os.listdir('cat_dog/train')
train_cat = filter(lambda x: x[:3] == 'cat', train_filenames)
train_dog = filter(lambda x: x[:3] == 'dog', train_filenames)


def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def process():
    rmrf_mkdir('train2')
    os.mkdir('train2/cat')
    os.mkdir('train2/dog')

    rmrf_mkdir('test2')
    os.symlink('cat_dog/test/', 'test2/test')

    for filename in train_cat:
        os.symlink('cat_dog/train/' + filename, 'train2/cat/' + filename)

    for filename in train_dog:
        os.symlink('cat_dog/train/' + filename, 'train2/dog/' + filename)


def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)

    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("train2", image_size, shuffle=False,
                                              batch_size=16)
    test_generator = gen.flow_from_directory("test2", image_size, shuffle=False,
                                             batch_size=16, class_mode=None)

    train = model.predict_generator(train_generator, len(train_generator.filenames), verbose=1)
    test = model.predict_generator(test_generator, len(test_generator.filenames), verbose=1)
    with h5py.File("gap_%s.h5" % str(MODEL)) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


write_gap(ResNet50, (224, 224))
write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
write_gap(Xception, (299, 299), xception.preprocess_input)
