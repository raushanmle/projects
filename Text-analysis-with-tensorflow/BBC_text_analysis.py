# Objective of this code : You'll learn working flow of text analysis.
# Author: Raushan Kumar

# Input data files are available in the same directory.
# importing standard libraries

import itertools
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow import keras
%matplotlib inline

# This code was tested with TensorFlow v1.8
# print("You have TensorFlow version", tf.__version__)

# ## Get the data
# The data has been exported from BigQuery and is stored on GCS, and can be access directly via 
#   `!gsutil cp gs://dataset-uploader/bbc/bbc-text.csv .`
# 
# It can also be accessed via https://storage.googleapis.com/dataset-uploader/bbc/bbc-text.csv 
# reading data
data = pd.read_csv("bbc_text.csv")

# Doing Some EDA to understand data
data['category'].value_counts()
train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

# ## Data preparation
# There's some work to be done in order for our data to be ready for training.
# 1. First we'll split the data into training and test sets.
# 1. Then we'll tokenize the words (text), and then convert them to a numbered index. 
# 1. Next we'll do the same for the labels (categories), by using the `LabelEncoder` utility.
# 1. Finally, we'll convert the labels to a one-hot representation.

train_cat, test_cat = train_test_split(data['category'], train_size)
train_text, test_text = train_test_split(data['text'], train_size)

max_words = 1000
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)

tokenize.fit_on_texts(train_text) # fit tokenizer to our training text data
x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_cat)
y_train = encoder.transform(train_cat)
y_test = encoder.transform(test_cat)

# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

# ## Train the model
# Build the model using Keras layers and hyperparameters of your choosing. Then call `model.fit()`
# This model trains very quickly and 2 epochs are already more than enough
# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data

batch_size = 32
epochs = 3
drop_ratio = 0.5

layers = keras.layers
models = keras.models
# Build the model
model = models.Sequential()
model.add(layers.Dense(512, input_shape=(max_words,)))
model.add(layers.Activation('relu'))
# model.add(layers.Dropout(drop_ratio))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

# ## Evaluate the model
# Evaluation is easy. Just call `model.evaluate()`.


# Evaluate the accuracy of our trained model
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# ## Hyperparameter tuning
# This is a good time to go back and tweak some parameters such as `epoch`, `batch size`, `dropout ratio`, network structure, activation function, and others, to see if you can improve the accuracy.
# 
# In this particular case, to make it more challenging, I recommend reducing the max words of the call to `keras.preprocessing.text.Tokenizer`. This will reduce the number of words for each input sample, thus making it more challenging to accurately predict the category. (Notice that not all hyperparameters are necessarily inside the model. This is one such example.)
# 
# The default was up to 1000 words per article. See what happens when you reduce that number to 200 words, or 50 words, or even fewer. As the evaluation accuracy drops, the effects of your hyperparameter tuning will be more pronounced, with successful adjustments making meaningful improvements to the model performance.
# 
# To make this process easier to manage, I've encapulated the model definition and training and evaluation calls into one function call. You can add additional parameters as needed.

def run_experiment(batch_size, epochs, drop_ratio):
    print('batch size: {}, epochs: {}, drop_ratio: {}'.format(
        batch_size, epochs, drop_ratio))
    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(max_words,)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(drop_ratio))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.1)
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print('\tTest loss:', score[0])
    print('\tTest accuracy:', score[1])

batch_size = 16
epochs = 4
drop_ratio = 0.4
run_experiment(batch_size, epochs, drop_ratio)

# ###Hyperparameter Search
# 
# You can also automate this process using for-loops and more sophiscated methods of deciding which combinations of hyperparameter values to try out.
# 
# Exhaustive search is generally not the most elegant way, this is mostly just for illustrative purposes.


# Note: The data processed for this output had the max text length set to 400.
# for batch_size in range(10,31,10):
#   for epochs in range(3,15,5):
#     for drop_ratio in np.linspace(0.1, 0.5, 3):
#       run_experiment(batch_size, epochs, drop_ratio)
# ## Make some predictions
# Take some samples from the test dataset and inspect some individual predictions, to ensure that things are sensible.
# 

# Here's how to generate a prediction on individual examples
text_labels = encoder.classes_

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_text.iloc[i][:50], "...")
    print('Actual label:' + test_cat.iloc[i])
    print("Predicted label: " + predicted_label + "\n")  

# ## (optional) Extra extra! Visualize the confusion matrix
# This can help identify which areas were a challenge to get right, if the model is performing poorly.

y_softmax = model.predict(x_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(y_test)):
    probs = y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)


# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)


cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(24,20))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
plt.show()
