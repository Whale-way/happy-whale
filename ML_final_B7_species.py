import numpy as np 
import pandas as pd 
import os
#allow to work on hertie server
os.environ['CUDA_DEVICE_ORDER'] ="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="2"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_HOME']= '/workspace/cache'
import gc
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from tqdm.autonotebook import tqdm
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import keras.backend as K
from keras.models import Sequential
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from keras.models import load_model
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

train_df = pd.read_csv("../happy-whale/input/train.csv")
train_df.head()
train_df_small = train_df[:50]


train_df.species = train_df.species.str.replace('kiler_whale','killer_whale')
train_df.species = train_df.species.str.replace('bottlenose_dolpin','bottlenose_dolphin')
train_df['species'][(train_df['species'] =="pilot_whale") | (train_df['species'] =="globis" )]='short_finned_pilot_whale'
animal_cnt = train_df.species.value_counts()
specs = list(animal_cnt.keys())
values = list(animal_cnt.values)
cmap = cm.get_cmap('jet')
norm = Normalize(vmin=0,vmax=len(specs))
cols = np.arange(0,len(specs))
train_jpg_path = "../happy-whale/input/cropped_train_images/cropped_train_images/"
test_jpg_path = "../happy-whale/input/cropped_test_images/cropped_test_images/"
train_images_list = os.listdir('../happy-whale/input/cropped_train_images/cropped_train_images')


def Loading_Images(data, m, dataset):
    print("Loading images")
    X_train = np.zeros((m, 32, 32, 3))
    count = 0
    for fig in tqdm(data['image']):
        img = image.load_img("../happy-whale/input"+dataset+"/"+fig, target_size=(32, 32, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x
        count += 1
    return X_train

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y = onehot_encoded
    return y, label_encoder

X = Loading_Images(train_df, train_df.shape[0], "/cropped_train_images/cropped_train_images")
X /= 255

y, label_encoder = prepare_labels(train_df['species'])

gc.collect()

from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

B7_model = EfficientNetB7(input_shape=(32,32,3), weights=None, include_top=False)

layer = B7_model.output
#layer = GlobalAveragePooling2D()(layer)#extra
#layer = Dropout(0.5)(layer)#extra
layer = Dense(1024, activation='relu')(layer)
#layer = Dense(512, activation='relu')(layer)#extra
layer = Flatten()(layer)
predictions = Dense(y.shape[1], activation='softmax')(layer)
model = Model(inputs=B7_model.input, outputs=predictions)

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

train_datagen = ImageDataGenerator(horizontal_flip=True,
                                   vertical_flip=True,
                                   validation_split=0.20,
                                   )

gc.collect()

history = model.fit(X, y, epochs=200, batch_size=2, verbose=1)

model.save('./effb7_0_species.h5')

def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (6, 6), strides = (1, 1), input_shape = (32, 32, 3)))
    model.add(BatchNormalization(axis = 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
      
    model.add(Conv2D(64, (3, 3), strides = (1,1)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3)))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.85))

    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    return(model)

Cnn_model = cnn_model()

del X
del y
gc.collect()

test = os.listdir("../happy-whale/input/cropped_test_images/cropped_test_images/")

col = ['image']
test_df = pd.DataFrame(test, columns=col)
test_df['predictions'] = ''

model = load_model(r'../happy-whale/effb7_0_species.h5')


batch_size=5000
batch_start = 0
batch_end = batch_size
L = len(test_df)

while batch_start < L:
    limit = min(batch_end, L)
    test_df_batch = test_df.iloc[batch_start:limit]
    print(type(test_df_batch))
    X = Loading_Images(test_df_batch, test_df_batch.shape[0], "/cropped_test_images/cropped_test_images")
    X /= 255
    predictions = model.predict(np.array(X), verbose=1)
    for i, pred in enumerate(predictions):
        p=pred.argsort()[-5:][::-1]
        idx=-1
        s=''
        s1=''
        s2=''
        for x in p:
            idx=idx+1
            if pred[x]>0.5:
                s1 = s1 + ' ' +  label_encoder.inverse_transform(p)[idx]
            else:
                s2 = s2 + ' ' + label_encoder.inverse_transform(p)[idx]
        s= s1 + ' new_species' + s2
        s = s.strip(' ')
        test_df.loc[ batch_start + i, 'predictions'] = s
    batch_start += batch_size   
    batch_end += batch_size
    del X
    del test_df_batch
    del predictions
    gc.collect()
test_df.to_csv('submissionb7_species.csv',index=False)


