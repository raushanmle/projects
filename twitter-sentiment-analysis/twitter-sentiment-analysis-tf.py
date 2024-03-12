
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r"C:\Users\raush\Downloads\twitter.csv", encoding='latin', names=[
                   'target', 'id', 'date', 'query', 'username', 'content'])

# Missing data is common occurance in datasets, therefore it is recommended to check if a data set contains missing values before starting any analysis.
data.isna().sum().sort_values(ascending=False)
# Changing labels from 0 and 4 for more informative labels for further analysis.
data['target'] = data['target'].replace([0, 4], ['Negative', 'Positive'])

# There are a lot of unnecessary columns in the following dataset. The task is to classify the semantics of a tweet, so all columns except the target and content columns are unnecessary.

data.drop(['id', 'date', 'query', 'username'], axis=1, inplace=True)
# Replacing Positive and Negative labels with 1 and 0 respectively.
data.target = data.target.replace({'Positive': 1, 'Negative': 0})

# ### Content cleaning
# Stemming - it does refers to the process which goal is to reduce words into thier base form. In case of our problem for classification it is very important ooperation as we need to focus on the meaning of particular word. For instance words: *Running, Runned, Runner* all can reduce to the stem *Run*. Below we have used the base of english stopwords and stemming algorithm from nltk library.

english_stopwords = stopwords.words('english')
# based on english stopwords
stemmer = SnowballStemmer('english')
# stemming algorithm
regex = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


def preprocess(content):
    content = re.sub(regex, ' ', str(content).lower()).strip()
    tokens = []
    for token in content.split():
        if token not in english_stopwords:
            tokens.append(stemmer.stem(token))
    return " ".join(tokens)


data.content = data.content.apply(lambda x: preprocess(x))

# ### Train test split
train, test = train_test_split(data, test_size=0.1, random_state=44)

# ### Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train.content)
vocab_size = len(tokenizer.word_index) + 1
max_length = 50
sequences_train = tokenizer.texts_to_sequences(train.content)
sequences_test = tokenizer.texts_to_sequences(test.content)

X_train = pad_sequences(sequences_train, maxlen=max_length, padding='post')
X_test = pad_sequences(sequences_test, maxlen=max_length, padding='post')
y_train = train.target.values
y_test = test.target.values

# ### Word embeddings using GloVe
# Word embeddings provide a dense representation of words and their relative meanings. Embedding Matrix is a maxtrix of all words and their corresponding embeddings. Embedding matrix is used in embedding layer in model to embedded a token into it's vector representation, that contains information regarding that token or word.
# Embedding vocabulary is taken from the tokenizer and the corresponding vectors from embedding model, which in this case is GloVe model. GloVe stand for Global Vectors for Word Representation
# and it is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.
# Below was used pretrained GloVe embeddings from world known Stanford vector files. The smallest available file contains embeddings created for tiny 6 billions of tokens.

embeddings_dictionary = dict()
embedding_dim = 100
glove_file = open(r'C:\Users\raush\Downloads\glove.6B.100d.txt')

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions

glove_file.close()

embeddings_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embeddings_matrix[index] = embedding_vector


# # Model test harness
# The proposed model architecture will be tested on the following parameters:
#
# *   loss = "binnary_crossentropy" (due to binary classification problem)
# *   optimizer = Adam(learning_rate=0.001) (may be changed after seeing the learning graph)
# *   metrics = "accuracy" (due to binary classification problem)
# *   number of epochs = 10 (due to the large training data set)
# *   batch size = 1000 (in order to accelerate learning time)


embedding_layer = tf.keras.layers.Embedding(
    vocab_size, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False)


num_epochs = 10
batch_size = 1000

# # Model - Embedding + Stacked LSTM
# Model consisted of layers build with lstm cells. With such a large amount of data, the model is computationally complex making the training process take a while. Furthermore, model regularization layers will reduce the possible overfitting which was present in the simpler models tested.


model = Sequential([
    embedding_layer,
    tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Bidirectional(LSTM(128)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(loss='binary_crossentropy', optimizer=Adam(
    learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size,
                    epochs=num_epochs, validation_data=(X_test, y_test))

# model.save("my_model.keras")
# model = tf.keras.models.load_model("my_model.keras")

y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)

print(classification_report(y_test, y_pred))

# History for accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train accuracy', 'Test accuracy'], loc='lower right')
plt.show()
# History for loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train loss', 'Test loss'], loc='upper right')
plt.suptitle('Accuracy and loss for second model')
plt.show()
