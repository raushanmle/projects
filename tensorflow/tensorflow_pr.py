import tensorflow as tf
import numpy as np
import keras


model = tf.keras.models.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))















# GRADED FUNCTION: training_dataset

def training_dataset():
    """Creates the training dataset out of the training images. Pixel values should be normalized.

    Returns:
        tf.data.Dataset: The dataset including the images of happy and sad faces.
    """
    
    ### START CODE HERE ###

    # Specify the function to load images from a directory and pass in the appropriate arguments:
    # - directory: should be a relative path to the directory containing the data. 
    #              You may hardcode this or use the previously defined global variable.
    # - image_size: set this equal to the resolution of each image (excluding the color dimension)
    # - batch_size: number of images the generator yields when asked for a next batch. Set this to 10.
    # - class_mode: How the labels are represented. Should be one of "binary", "categorical" or "int".
    #               Pick the one that better suits here given that the labels can only be two different values.
    train_dataset = None(
        directory=None,
        image_size=None,
        batch_size=None,
        label_mode=None
    )

    # Define the rescaling layer (passing in the appropriate parameters)
    rescale_layer = None

    # Apply the rescaling by using the map method and a lambda
    train_dataset_scaled = None
    
    ### END CODE HERE ###

    return train_dataset_scaled



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback

# GRADED FUNCTION: training_dataset

def training_dataset():
    """Creates the training dataset out of the training images. Pixel values should be normalized.

    Returns:
        tf.data.Dataset: The dataset including the images of happy and sad faces.
    """
    
    ### START CODE HERE ###

    # Specify the function to load images from a directory and pass in the appropriate arguments:
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory='happy_or_sad',  # Replace with the actual path to your dataset
        image_size=(128, 128),  # Replace with the actual image size
        batch_size=10,
        label_mode='binary'  # Assuming binary classification for happy and sad faces
    )

    # Define the rescaling layer
    rescale_layer = tf.keras.layers.Rescaling(1./255)

    # Apply the rescaling by using the map method and a lambda
    train_dataset_scaled = train_dataset.map(lambda x, y: (rescale_layer(x), y))
    
    ### END CODE HERE ###

    return train_dataset_scaled

# Load the training dataset
train_dataset = training_dataset()

# Define a custom callback to stop training when 99.9% accuracy is reached
class StopTrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the custom callback
model.fit(train_dataset, epochs=20, callbacks=[StopTrainingCallback()])



BASE_DIR = "./data/"
happy_dir = os.path.join(BASE_DIR, "happy/")
sad_dir = os.path.join(BASE_DIR, "sad/")

fig, axs = plt.subplots(1, 2, figsize=(6, 6))
axs[0].imshow(tf.keras.utils.load_img(f"{os.path.join(happy_dir, os.listdir(happy_dir)[0])}"))
axs[0].set_title('Example happy Face')

axs[1].imshow(tf.keras.utils.load_img(f"{os.path.join(sad_dir, os.listdir(sad_dir)[0])}"))
axs[1].set_title('Example sad Face')

plt.tight_layout()

import tensorflow as tf
import os

# GRADED FUNCTION: training_dataset

def training_dataset():
    """Creates the training dataset out of the training images. Pixel values should be normalized.

    Returns:
        tf.data.Dataset: The dataset including the images of happy and sad faces.
    """
    
    ### START CODE HERE ###

    # Specify the function to load images from a directory and pass in the appropriate arguments:
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory='./data',  # Replace with the actual path to your dataset
        image_size=(128, 128),  # Replace with the actual image size
        batch_size=10,
        label_mode='binary'  # Assuming binary classification for happy and sad faces
    )

    # Define the rescaling layer
    rescale_layer = tf.keras.layers.Rescaling(1./255)

    # Apply the rescaling by using the map method and a lambda
    train_dataset_scaled = train_dataset.map(lambda x, y: (rescale_layer(x), y))
    
    ### END CODE HERE ###

    return train_dataset_scaled

# Load the training dataset
train_dataset = training_dataset()











