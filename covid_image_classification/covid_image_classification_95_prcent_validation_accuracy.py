
############################## Importing Libraries #################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers import Input, Add, Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
import keras
from keras.callbacks import Callback
from keras.models import Model
from keras.preprocessing.image import array_to_img
from PIL import Image
from numpy import asarray
import os
%matplotlib inline


# Setting path variables
path = "../input/covid19-image-dataset/Covid19-dataset/train"
path_test = "../input/covid19-image-dataset/Covid19-dataset/test"


def min_size(path_of_image):
    """
    Checking min size available in all folder.
    This will help you to give idea, which size you have to select for resizing.
    
    """
    # wlking across directory to open all available images
    size_images = {}
    for dirpath, _, filenames in os.walk(path_of_image):
        for path_image in filenames:
            image = os.path.abspath(os.path.join(dirpath, path_image))
            
            # checking images size and storing in a dict to compare
            with Image.open(image) as img:
                width, heigth = img.size
                size_images[path_image] = {'width': width, 'heigth': heigth}
                
    # Creating a small DF to check min & max size of images
    df_image_desc = pd.DataFrame(size_images).T
    min_width = df_image_desc['width'].min()
    min_height = df_image_desc['heigth'].min()
    
    return min_height, min_width

min_size(path)



################################ Data prep started #################################

def load_image_from_folder(path, basewidth, hsize):
    
    """
    Loading all images in a numpy array with labels
    
    """
    # creating temp array
    image_array = []
    labels = []
    # directory walking started
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file != []:
                # trying to get path of each images
                path_updated = os.path.join(subdir, file)
                # fetching lables from directory names
                label = subdir.split("/")[-1]
                labels.append(label)
                # Converting image & resizing it
                img = Image.open(path_updated).convert('L')
                img = img.resize((basewidth, hsize), Image.ANTIALIAS)
                frame = asarray(img)
                # appending array of image in temp array
                image_array.append(frame)
                
    # Now i have to convert this images to array channel format which can be done using zero matrix
    # creating a dummy zero matrix of same shape with single channel
    
    image_array1 = np.zeros(shape=(np.array(image_array).shape[0], hsize, basewidth,  1))
    for i in range(np.array(image_array).shape[0]):
        # finally each sub matrix will be replaced with respective images array
        image_array1[i, :, :, 0] = image_array[i]
    
    return image_array1, np.array(labels)

image_array, labels = load_image_from_folder(path, 48, 48)
image_array_test, labels_test = load_image_from_folder(path_test, 48, 48)



def vis_training(hlist, start=1):
    
    """
    This function will help to visualize the loss, val_loss, accuracy etc.
    
    """
    # getting history of all kpi for each epochs
    loss = np.concatenate([hlist.history['loss']])
    val_loss = np.concatenate([hlist.history['val_loss']])
    acc = np.concatenate([hlist.history['accuracy']])
    val_acc = np.concatenate([hlist.history['val_accuracy']])
    epoch_range = range(1,len(loss)+1)
    
    # Block for training vs validation loss
    plt.figure(figsize=[12,6])
    plt.subplot(1,2,1)
    plt.plot(epoch_range[start-1:], loss[start-1:], label='Training Loss')
    plt.plot(epoch_range[start-1:], val_loss[start-1:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    # Block for training vs validation accuracy
    plt.subplot(1,2,2)
    plt.plot(epoch_range[start-1:], acc[start-1:], label='Training Accuracy')
    plt.plot(epoch_range[start-1:], val_acc[start-1:], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


# Since this is a small dataset so i need to generate some data increase accuracy
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images



# Converting labels in int format as TF accepts only int in targets class
labels[labels == 'Covid'] = 0
labels[labels == 'Normal'] = 1
labels[labels == 'Viral Pneumonia'] = 2

labels_test[labels_test == 'Covid'] = 0
labels_test[labels_test == 'Normal'] = 1
labels_test[labels_test == 'Viral Pneumonia'] = 2
# Converting to int format
labels = labels.astype('int')
labels_test = labels_test.astype('int')

# Adding extra data from datagen
train_gen = datagen.flow(image_array, labels)
test_gen = datagen.flow(image_array_test, labels_test)

################################ Data Prep ended ###################################


################################ model building started ############################
def model_seq():
    """
    This decorator will help to create a CNN model.
    
    """
    # just clearing the session
    tf.keras.backend.clear_session()
    model=Sequential()

    # Adding Conv layer  
    model.add(Conv2D(filters=32, kernel_size = (3,3), activation="relu", input_shape=(48, 48, 1)))
    # Added max pooling layer
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Added normalization layer
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    
    # Added flatten option
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(128,activation="relu"))
    # Adding dropout of 20%
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(Dense(3, activation="softmax"))

    optimiser = tf.keras.optimizers.Adam()
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=optimiser, metrics=["accuracy"])
    # Printing summary of layers
    print(model.summary())

    return model



class TerminateOnBaseline(Callback):

    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='val_accuracy', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True

#  Setting callbacks i) Whenever it sees convergence reduce the learning rate ii) Once it reaches to desired val_accuracy
# terminates the training
callbacks = [
tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',min_lr=0.00001,factor=0.2, patience=1, verbose=1),
TerminateOnBaseline(monitor='val_accuracy', baseline=0.95)]


# Model training
model_covid = model_seq()

# Running model training
batch_size = 8
train_log = model_covid.fit(
    train_gen,
    validation_data = test_gen,
    epochs=200,
    batch_size = batch_size,
    callbacks = callbacks
    )


# Visuals of loss and accuracy
vis_training(train_log, start=1)


