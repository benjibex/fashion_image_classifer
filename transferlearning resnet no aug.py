# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:45:37 2018

@author: Benjibex
"""

#based on the tutorial by Satya Mallick https://github.com/spmallick/learnopencv/blob/master/Keras-Transfer-Learning/transfer-learning-vgg.ipynb
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from __future__ import print_function
import keras
from keras.utils import to_categorical
import os
from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.applications import resnet50

res_conv = resnet50.ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

res_conv.summary()

train_dir = './Documents/ML_Project/fashion/train'
validation_dir = './Documents/ML_Project/fashion/valid'
test_dir = './Documents/ML_Project/fashion/test'

nTrain = 1920
nVal = 480
nTest = 600

#adding data augmentation, before it was the imagerescale only
datagen = ImageDataGenerator(rescale=1./255) #rescale images to be between 0 and 1
batch_size = 20
 
train_features = np.zeros(shape=(nTrain, 7, 7, 2048))
train_labels = np.zeros(shape=(nTrain,3))
 
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle='shuffle')


i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = res_conv.predict(inputs_batch)
    train_features[i * batch_size : (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break
         
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 2048))

validation_features = np.zeros(shape=(nVal, 7, 7, 2048))
validation_labels = np.zeros(shape=(nVal,3))

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = res_conv.predict(inputs_batch)
    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 2048))



from keras import models
from keras import layers
from keras import optimizers
 
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 2048))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])
 
history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels))

# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#examine the errors 
fnames = validation_generator.filenames
 
ground_truth = validation_generator.classes
 
label2index = validation_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())
 
predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)
 
errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nVal))

for i in range(len(errors)):
    pred_class = np.argmax(prob[errors[i]])
    pred_label = idx2label[pred_class]
     
    print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        prob[errors[i]][pred_class]))
     
    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.imshow(original)
    plt.show()

#make predictions i cant get this code to work here for some reason
#flow the test images through now so we can make predictions on the test data
test_features = np.zeros(shape=(nTest, 7, 7, 2048))
test_labels = np.zeros(shape=(nTest,3))

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in test_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    test_features[i * batch_size : (i + 1) * batch_size] = features_batch
    test_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTest:
        break

test_features = np.reshape(test_features, (nTest, 7 * 7 * 2048))

#make predictions on the test data

ground_truth_test = test_generator.classes
 
predictions_test = model.predict_classes(test_features)
prob_test = model.predict(test_features)


errors_test = np.where(predictions_test != ground_truth_test)[0]
print("No of test errors = {}/{}".format(len(errors_test),nTest))

