
from keras.layers import Input, Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import regularizers
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import Callback

train = pd.read_csv(
    "C:\\Code\\projects\\Emotion-classification\\train_data\\train_data.csv")


def vis_training(hlist, start=1):
    loss = np.concatenate([h.history['loss'] for h in hlist])
    val_loss = np.concatenate([h.history['val_loss'] for h in hlist])
    acc = np.concatenate([h.history['accuracy'] for h in hlist])
    val_acc = np.concatenate([h.history['val_accuracy'] for h in hlist])
    epoch_range = range(1, len(loss)+1)
    plt.figure(figsize=[12, 6])
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range[start-1:], loss[start-1:], label='Training Loss')
    plt.plot(epoch_range[start-1:], val_loss[start-1:],
             label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epoch_range[start-1:], acc[start-1:], label='Training Accuracy')
    plt.plot(epoch_range[start-1:], val_acc[start-1:],
             label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=10,
    zoom_range=0.1,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

def prepare_data(data, column_name):
    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data[column_name])))
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i, :, :, 0] = image / 255
    return image_array, image_label

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


full_train, full_train_labels = prepare_data(train, 'emotion')
train_images, valid_images, train_labels, valid_labels = train_test_split(
    full_train, full_train_labels, test_size=0.2, random_state=1)
train_gen = datagen.flow(train_images, train_labels)
test_gen = datagen.flow(valid_images, valid_labels)

def model_seq2():
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(150, activation="relu",
              kernel_regularizer=regularizers.l2(l2=.01)))
    model.add(Dense(50, activation="relu",
              kernel_regularizer=regularizers.l2(l2=.01)))
    model.add(Dense(7, activation="softmax"))
    optimiser = tf.keras.optimizers.SGD(learning_rate=.01)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimiser, metrics=["accuracy"])
    print(model.summary())
    return model

model = model_seq2()
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', min_lr=0.00001, factor=0.2, patience=5, verbose=1), TerminateOnBaseline(monitor='val_accuracy', baseline=0.9)]

batch_size = 10
train_log = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=100,
    callbacks=callbacks
)
