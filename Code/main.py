# Imports
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import EarlyStopping
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import Word
from scipy import stats
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Embedding, SpatialDropout1D, Dense, LSTM, Dropout


# Reading in the data & creating a single dataframe
colnames = ['review', 'rating']
yelp = pd.read_csv('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\yelp_labelled.txt',
                   sep='\t', header=None, encoding='latin-1', names=colnames)
imdb = pd.read_csv('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\imdb_labelled.txt',
                   sep='\t', header=None, names=colnames)
amzn = pd.read_csv('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\amazon_cells_labelled.txt',
                   sep='\t', header=None, names=colnames)

df = pd.concat((yelp, imdb, amzn), ignore_index=True)
pd.set_option('max_colwidth', 50)
pd.set_option('display.max_columns', 10)

print(df.info())
print(df.head(10), '\n')
print("Value counts of ratings:")
print(df['rating'].value_counts(), '\n')

# Cleaning review strings - removing non-english characters
def isEnglish(dataframe):
    notAsciiCount = 0
    for x in dataframe:
        if not x.isascii():
            notAsciiCount += 1
    return notAsciiCount

print("Count of reviews with non-English characters: ", isEnglish(df['review']), '\n')

df['review'] = df['review'].apply(lambda x: ''.join(char for char in x if ord(char) < 128))
df['review'] = df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
df['review'] = df['review'].str.lower()

print("New count of reviews with non-English characters: ", isEnglish(df['review']), '\n')

# Cleaning review strings - removing stopwords
stopwords = set(stopwords.words('english'))
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
print(df['review'].head(10), '\n')

# Cleaning review strings - lemmatizaiton
df['review'] = df['review'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))
print("Lemmatized")
print(df['review'].head(10), '\n')

# Determining dictionary size
# Borrowed Code Author: Usman Malik
# URL: https://stackabuse.com/python-for-nlp-word-embeddings-for-deep-learning-in-keras/
all_words = []
for x in df['review']:
    tokenize_word = word_tokenize(x)
    for word in tokenize_word:
        all_words.append(word)

unique_words = set(all_words)
max_words = len(unique_words)
print("Count of unique words:", max_words)
# End borrowed code

# Analyzing length of reviews to determine padding max length
df['length'] = df['review'].str.split().str.len()
print("Analyzing Review Length:")
print("Mean (rounded):", round(np.mean(df['length'])))
print("Median:", np.median(df['length']))
print("Mode:", stats.mode(df['length']), '\n')
print("The review with the most words has", df['length'].max(), "words.\n")
print("Value Counts of Length column:")
print(df['length'].value_counts(), '\n')

# Dropping rows with low word counts
print("Before dropping reviews of length 0:", df.shape)
df = df[df['length'] > 3]
print("After dropping reviews of length 0:", df.shape, '\n')

# Storing mean value to use as padding length & in embedding layer
mean = round(np.mean(df['length']))


# vectorizing & padding
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['review'])
seq = tokenizer.texts_to_sequences(df['review'])
print("Embedded sentences:")
print(seq[0:10], '\n')
review_vectors = pad_sequences(seq, padding='post', maxlen=mean)
print("Padded sentences:")
print(review_vectors[0:10], '\n')

ratings = list(df['rating'])

# Saving prepared data
np.savetxt('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\review_vectors.csv', review_vectors, delimiter=', ', fmt='% s')
np.savetxt('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\ratings.csv', ratings, delimiter=', ', fmt='% s')


# Train/Test Split
reviews_train, reviews_test, ratings_train, ratings_test = \
    train_test_split(review_vectors, ratings, test_size=0.3, random_state=22)

# Basic review of train/test split
print("Reviews Train:\n", reviews_train[0:10])
print("Length:", len(reviews_train))
print("Reviews Test:\n", reviews_test[0:10])
print("Length:", len(reviews_test))
print("Ratings Train:\n", ratings_train[0:10])
print("Length:", len(ratings_train))
print("Ratings Test:\n", ratings_test[0:10])
print("Length:", len(ratings_test))


# Model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_vector_length, input_length=mean))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(64, activation='LeakyReLU'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

early_stopping_monitor = EarlyStopping(patience=3)
history = model.fit(reviews_train, np.array(ratings_train), epochs=20, batch_size=128,
                    validation_data=(reviews_test, np.array(ratings_test)))

# loss, acc =
print(model.evaluate(reviews_test, np.array(ratings_test)))

# Visualizing accuracy & loss
epoch_range = range(1, 21)
plt.plot(epoch_range, history.history['accuracy'])
plt.plot(epoch_range, history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
plt.savefig('model_accuracy_plot.png')

plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Val'], loc='upper left')
plt.show()
plt.savefig('model_loss_plot.png')
