import numpy as np
import pandas as pd
import os
import warnings
from keras.layers import Dense, Flatten, Dropout, Lambda, Input, Concatenate, concatenate
from keras.models import Model
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)

train_filenames = os.listdir("cat_dog/train")
train_labels = []
for file in train_filenames:
    category = file.split('_')[0]
    if category == 'cat':
        train_labels.append('cat')
    else:
        train_labels.append('dog')

train_df = pd.DataFrame({
    'filename': train_filenames,
    'label': train_labels
})

valid_filenames = os.listdir("cat_dog/val")
valid_labels = []
for file in valid_filenames:
    category = file.split('_')[0]
    if category == 'cat':
        valid_labels.append('cat')
    else:
        valid_labels.append('dog')

validation_df = pd.DataFrame({
    'filename': valid_filenames,
    'label': valid_labels
})

batch_size = 48
train_num = len(train_df)
validation_num = len(validation_df)
print(validation_df)


def two_image_generator(generator, df, directory, batch_size,
                        x_col='filename', y_col=None, model=None, shuffle=False,
                        img_size1=(224, 224), img_size2=(299, 299)):
    gen1 = generator.flow_from_dataframe(
        df,
        directory,
        x_col=x_col,
        y_col=y_col,
        target_size=img_size1,
        class_mode=model,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=1)
    gen2 = generator.flow_from_dataframe(
        df,
        directory,
        x_col=x_col,
        y_col=y_col,
        target_size=img_size2,
        class_mode=model,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=1)

    while True:
        X1i = gen1.next()
        X2i = gen2.next()
        if y_col:
            yield [X1i[0], X2i[0]], X1i[1]  # X1i[1] is the label
        else:
            yield [X1i, X2i]


# add data_augmentation
train_aug_datagen = ImageDataGenerator(
    rotation_range=20,
    shear_range=0.1,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
train_generator = two_image_generator(train_aug_datagen, train_df, 'cat_dog/train',
                                      batch_size=batch_size, y_col='label',
                                      model='binary', shuffle=True)

validation_datagen = ImageDataGenerator()

validation_generator = two_image_generator(validation_datagen, validation_df,
                                           'cat_dog/val', batch_size=batch_size,
                                           y_col='label', model='binary', shuffle=True)


def create_base_model(MODEL, img_size, lambda_fun=None):
    inp = Input(shape=(img_size[0], img_size[1], 3))
    x = inp
    if lambda_fun:
        x = Lambda(lambda_fun)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False, pooling='avg')

    model = Model(inp, base_model.output)
    return model


# define vgg + resnet50 + densenet
model1 = create_base_model(vgg16.VGG16, (224, 224), vgg16.preprocess_input)
model2 = create_base_model(resnet50.ResNet50, (224, 224), resnet50.preprocess_input)
model3 = create_base_model(inception_v3.InceptionV3, (299, 299), inception_v3.preprocess_input)
model1.trainable = False
model2.trainable = False
model3.trainable = False

inpA = Input(shape=(224, 224, 3))
inpB = Input(shape=(299, 299, 3))
out1 = model1(inpA)
out2 = model2(inpA)
out3 = model3(inpB)

x = Concatenate()([out1, out2, out3])
x = Dropout(0.6)(x)
x = Dense(1, activation='sigmoid')(x)
multiple_pretained_model = Model([inpA, inpB], x)


def train():
    multiple_pretained_model.compile(loss='binary_crossentropy',
                                     optimizer='rmsprop',
                                     metrics=['accuracy'])

    multiple_pretained_model.summary()

    checkpointer = ModelCheckpoint(filepath='dogcat.weights.best.hdf5', verbose=1,
                                   save_best_only=True, save_weights_only=True)

    multiple_pretained_model.fit_generator(
        train_generator,
        epochs=5,
        steps_per_epoch=train_num // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_num // batch_size,
        verbose=1,
        callbacks=[checkpointer]
    )


def predict():
    print("正在加载模型：")
    multiple_pretained_model.load_weights('dogcat.weights.best.hdf5')
    test_filenames = os.listdir("cat_dog/test")
    test_df = pd.DataFrame({
        'filename': test_filenames
    })
    print(test_df)
    test_df['id'] = test_df['filename'].apply(lambda x: int(x.split('.')[0]))
    num_test = len(test_df)
    test_datagen = ImageDataGenerator()
    test_generator = two_image_generator(test_datagen, test_df, 'cat_dog/test', batch_size=batch_size)

    prediction = multiple_pretained_model.predict_generator(test_generator,
                                                            steps=np.ceil(num_test / batch_size),
                                                            # workers=8,
                                                            verbose=1)
    prediction = prediction.clip(min=0.005, max=0.995)

    print(prediction)
    print(type(prediction))
    print(list(prediction))
    print(prediction.tolist())
    res=prediction.tolist()[0]
    res = [num[0] for num in res]
    test_df['pred'] =res
    print(test_df)
    test_df['label'] = test_df['pred'].apply(lambda x: 1 if x > 0.5 else 0)
    # test_df['label'] = np.argmax(prediction, axis=1)
    test_df[['id', 'label']].to_csv('pretrained.csv', index=None, header=False)


predict()
