# There are 10,000 data in the training set and 1,000 data in the testing set. Each data in the training/testing set consists of 3 components:
# > - Story - consists of single or multiple sentences
# > - Question - single sentence query related to the story
# > - Answer - "yes" or "no" answer to the question

# ## Data Preprocessing

import pickle
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
%matplotlib inline

# Read as binary
with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)

# Read as binary
with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)

print("Length of the train data: ", len(train_data))
print("Length of the test data: ", len(test_data))

all_data = test_data + train_data

# Build vocabulary from all stories and questions
vocab = set()

for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

vocab.add('no')
vocab.add('yes')

# Add one to length of vocabulary: Keras embedding layer requires this.
vocab_len = len(vocab) + 1
print("Actual length of the vocabulary: ", vocab_len-1)

# Length of all the stories
all_story_len = [len(data[0]) for data in all_data]
# Get maximum of the stories
max_story_len = max(all_story_len)
max_question_len = max([len(data[1]) for data in all_data])

print("Maximum length of the stories: ", max_story_len)
print("Maximum length of the question: ", max_question_len)

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

#tokenizer.word_index
train_story_text = []
train_question_text = []
train_answers = []

for story, question, answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(answer)

# Train_story_text is a list of lists of words
train_story_seq = tokenizer.texts_to_sequences(train_story_text)

print(len(train_story_seq))
print(len(train_story_text))

# Create our own list of list of word indicies with padding.
def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len, max_question_len=max_question_len):
    # Stories = X
    X = []
    # Questions = Xq
    Xq = []
    # Y Correct Answer ['yes', 'no']
    Y = []
    for story, query, answer in data:
        # for each story
        # [23, 14, 15]
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        y = np.zeros(len(word_index)+1)
        y[word_index[answer]] = 1
        X.append(x)   # X holds list of lists of word indices for stories.
        Xq.append(xq) # Xq holds list of lists for word indices for questions.
        Y.append(y) # Y holds lists of lists of (38) biniary numbers, only 1 of them is 1.
        
    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))


inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)

# tokenizer.word_index['yes']
# tokenizer.word_index['no']
# 497 of the questions have answer 'yes', 503 of the questions have answer 'no'.
sum(answers_test)

# PLACEHOLDER shape=(max_story_len, batch_size)
input_sequence = Input((max_story_len,))
question = Input((max_question_len,))
# vocab_len
vocab_size = len(vocab) + 1
# INPUT ENCODER M
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
input_encoder_m.add(Dropout(0.3))

# OUTPUT
# (samples, story_maxlen, embedding_dim)
# INPUT ENCODER C
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))

# OUTPUT
# (samples, story_maxlen, max_question_len)

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_question_len))
question_encoder.add(Dropout(0.3))

# OUTPUT
# (samples, query_maxlen, embedding_dim)

# ENCODED <---- ENCODER(INPUT)
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

# input_encoded_m: (batch_size, story_maxlen, embedding_dim)
# input_encoded_c: (batch_size, story_maxlen, query_maxlen)
# question_encoded: (batch_size, query_maxlen, embedding_dim)


print(input_encoded_m.shape)
print(question_encoded.shape)


match = dot([input_encoded_m, question_encoded], axes=(2,2)) # why axes is (2,2) ==> dot product along the embedding dim (64 numbers dot 64 numbers)
match = Activation('softmax')(match)

# NOTE: match after dot: (batch_size, story_maxlen, query_maxlen)
# match after Activation: (batch_size, story_maxlen, query_maxlen)
# **NOTE: match (after dot): (batch_size, story_maxlen, query_maxlen)!!!**


response = add([match, input_encoded_c]) # (samples, story_maxlen, query_maxlen)
response = Permute((2,1))(response) # (samples, query_maxlen, story_maxlen)

# response after add: (batch_size, story_maxlen, query_maxlen)
# response after Permute: (batch_size, query_maxlen, story_maxlen)
# **NOTE: response after Permute: (batch_size, query_maxlen, story_maxlen)!!!**
answer = concatenate([response, question_encoded])

# Note: answer: (batch_size, query_maxlen, story_maxlen+embedding_dim)
# **NOTE: answer (after concatenate): (batch_size, query_maxlen, story_maxlen+embedding_dim)!!!**
answer = LSTM(32)(answer) #(samples, 32)
# answer: (batch_size, 32)
print(answer.shape)
answer = Dropout(0.5)(answer)
# answer: (batch_size, 32)
answer = Dense(vocab_size)(answer) # (samples, vocab_size) # YES/NO 0000
# answer (batch_size, vocab_size)
answer = Activation('softmax')(answer)
# answer: (batch_size, vocab_size)

model = Model([input_sequence, question], answer)

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit([inputs_train, queries_train], answers_train, batch_size=32, epochs=100, validation_data=([inputs_test, queries_test], answers_test))



print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('trainedmodel.h5')

pred_result = model.predict(([inputs_test, queries_test]))
pred_result.shape
pred_result[0]

index_word = {index: word for word, index in tokenizer.word_index.items()}


predictions = np.argmax(pred_result, axis=1)
pred_answers = [index_word[pred] for pred in predictions]
# ## Test with a run time question.
my_story = "John left the kitchen . Sandra dropped the football in the garden ."
my_story.split()
my_question = "Is the football in the garden ?"
my_question.split()

# The answer should be 'yes'.
mydata = [(my_story.split(), my_question.split(), 'yes')]

my_story, my_ques, my_ans = vectorize_stories(mydata)
pred_result = model.predict([my_story, my_ques])

val_max = np.argmax(pred_result[0])

# Get the answer corresponding to the highest predict probabilty.
for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

# Find out what's the highest predict probability.
pred_result[0][val_max]


