
!pip install tensorflow-gpu

import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

!pip list

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

import cv2
import imghdr

data_dir = 'data'

image_exts = ['jpeg','jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

data_dir="/content/sample_data/Data"

dir(data_dir)

import os
os.listdir(data_dir)

str="hello1"
str.isalnum()

str1="newfile.txt"

if str1.endswith(".txt"):

import os

# Specify the directory path
directory = "/content/sample_data/data"

# List all files/subdirectories
files = os.listdir.join(directory,'happy')
print("Files in sample_data:", files)

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

import os

# Specify the directory path
directory = "/content/sample_data/data"
for image_class in os.listdir(directory):
  for image in os.listdir(os.path.join(directory,image_class)):
    image_path=os.path.join(directory,image_class,image)
    try:
      img=cv2.imread(image_path)
      tip=imghdr.what(image_path)
      if tip not in image_exts:
        print("image not in ext list {}".format(image_path))
        os.remove(image_path)
    except Exception as e:
      print("Issue with image {}".format(image_path))

tf.data.Dataset??

import numpy as np
import matplotlib.pyplot as plt

data=tf.keras.utils.image_dataset_from_directory("/content/sample_data/data")

data

data_iterator = data.as_numpy_iterator()
data_iterator

batch = data_iterator.next()

batch

batch[0].shape

batch[1]

fig,ax = plt.subplots(ncols=4,figsize=(20,20))
for idx,img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])



data = data.map(lambda x, y:(x/255,y))

scaled_iter = data.as_numpy_iterator()

batch =scaled_iter.next()
batch[1].min()

fig,ax = plt.subplots(ncols=4,figsize=(20,20))
for idx,img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

len(data)

train_size+test_size+val_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

len(test)

"""Model"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense,Flatten,Dropout

model = Sequential()

model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3),1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])

model.summary()

logdir = "/content/sample_data/logs"

tensorboard_callback =  tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train,epochs=20,validation_data=val, callbacks=[tensorboard_callback])

hist.history

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal',label='val_loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
fig.suptitle('Loss',fontsize=20)
plt.legend(loc="upper left")
plt.show()

