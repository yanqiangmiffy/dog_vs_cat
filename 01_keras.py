from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
import shutil
import pandas as pd


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


process()
# dimensions of our images.
img_width, img_height = 200, 200
path = ''
# path = 'D://Project//dog_vs_cats//'
train_data_dir = path + 'train2'
validation_data_dir = path + 'valid2'
test_data_dir = path + 'test2'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 10
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


def train():
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = valid_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')

train()
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='categorical', shuffle=False)
model.load_weights('first_try.h5')

sub = pd.DataFrame()
filenames = test_generator.filenames
nb_samples = len(filenames)

ids = [int(f.split('.')[0][5:]) for f in filenames]
preds = model.predict_generator(test_generator, steps=nb_samples)
sub['id'] = ids
sub['pred'] = preds
sub['label'] = sub['pred'].apply(lambda x: 1 if x > 0.5 else 0)
sub[['id', 'label']].to_csv('keras_cnn.csv', index=None, header=False)
