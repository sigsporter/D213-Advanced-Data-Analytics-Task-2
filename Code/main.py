# Imports
import math
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import Word
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer, one_hot
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Embedding, SpatialDropout1D, Dense, LSTM, Dropout, Flatten

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
def isEnglish(dataframe):
    notAsciiCount = 0
    for x in dataframe:
        if not x.isascii():
            notAsciiCount += 1
    return notAsciiCount


# print("Count of reviews with non-English characters: ", isEnglish(df['review']), '\n')

df['review'] = df['review'].apply(lambda x: ''.join(char for char in x if ord(char) < 128))
df['review'] = df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
df['review'] = df['review'].str.lower()

# print("New count of reviews with non-English characters: ", isEnglish(df['review']), '\n')
print("Value counts of ratings:")
print(df['rating'].value_counts(), '\n')

stopwords = set(stopwords.words('english'))
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
print(df['review'].head(10), '\n')



# wordnet_lemmatizer = WordNetLemmatizer()
# def lemmatizer(text):
#     lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
#     return lemm_text

df['review'] = df['review'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))
print("Lemmatized")
print(df['review'].head(10), '\n')

# cleaned = RegexpTokenizer('\w+')
# df['review'] = df['review'].apply(cleaned.tokenize)
# print(df['review'].head(10), '\n')

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

df['length'] = df['review'].str.split().str.len()
longest_review = df['length'].max()
print("The review with the most words has", longest_review, "words.\n")

mean = math.ceil(df['length'].mean())
print(mean)

# padding & vectorizing
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['review'])
seq = tokenizer.texts_to_sequences(df['review'])
print("Embedded sentences:")
print(seq[0:10], '\n')
review_vectors = pad_sequences(seq, padding='post', maxlen=mean)
print("Padded sentences:")
print(review_vectors[0:10], '\n')


# Train/Test Split
ratings = list(df['rating'])

reviews_train, reviews_test, ratings_train, ratings_test = train_test_split(review_vectors, ratings, test_size=0.3,
                                                                            random_state=22)

print("Reviews Train:\n", reviews_train[0:10])
print("Length:", len(reviews_train))
print("Reviews Test:\n", reviews_test[0:10])
print("Length:", len(reviews_test))
print("Ratings Train:\n", ratings_train[0:10])
print("Length:", len(ratings_train))
print("Ratings Test:\n", ratings_test[0:10])
print("Length:", len(ratings_test))

# np.savetxt('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\review_vextors_2.csv', review_vectors, delimiter=', ', fmt='% s')
# np.savetxt('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\ratings_2.csv', ratings, delimiter=', ', fmt='% s')

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

# # loss1, acc1, mse1 =
# # print(model.evaluate(reviews_test, np.array(ratings_test)))
history = model.fit(reviews_train, np.array(ratings_train), epochs=5,
                    validation_data=(reviews_test, np.array(ratings_test)))


'''
df['word_count'] = [len(x.split()) for x in df['review']]
df['char_count'] = df['review'].apply(len)
print(df)
print(df['rating'].value_counts(), '\n')
'''
'''
print("Max word count: ", max(len(x.split()) for x in df['review']), '\n')

# Review Cleanup
df['review'] = df['review'].astype(str).str.lower()
df['review_clean'] = df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
print(df['review_clean'], '\n')

# stopwords
stopwords = set(stopwords.words('english'))

df['no_stop'] = df['review_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
print(df['no_stop'].head(10), '\n')

# tokenize & lemmatize
cleaned = RegexpTokenizer('\w+')
df['tok'] = df['no_stop'].apply(cleaned.tokenize)
print(df['tok'].head(10), '\n')

wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

df['lem'] = df['tok'].apply(lambda x: lemmatizer(x))
print("Stopwords Removed & Lemmatized")
print(df['lem'].head(10), '\n')

# wordcloud
def lem_string(list):
    string = ' '.join(list)
    return string

df['lem_string'] = df['lem'].apply(lambda x: lem_string(x))
print(df['lem_string'].head(10))

text = "".join(x for x in df['lem_string'])
wordcloud = WordCloud(stopwords=stopwords).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("wordcloud1.png")
# plt.show()

max_words = max(len(x.split()) for x in df['no_stop'])
print("Max word count: ", max_words, '\n')

# padding & vectorizing
tokenizer = Tokenizer(num_words = max_words+1)
tokenizer.fit_on_texts(df['lem'])
seq = tokenizer.texts_to_sequences(df['lem'])
print(seq[0:10], '\n')

review_vectors = pad_sequences(seq, padding='post', maxlen=max_words)
print(review_vectors[0:10])

# Train/Test Split
ratings = list(df['rating'])

reviews_train, reviews_test, ratings_train, ratings_test = train_test_split(review_vectors, ratings, test_size=0.2,
                                                                            random_state=22)

print("Reviews Train:\n", reviews_train[0:10])
print("Reviews Test:\n", reviews_test[0:10])
print("Ratings Train:\n", ratings_train[0:10])
print("Ratings Test:\n", ratings_test[0:10])

np.savetxt('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\review_vextors.csv', review_vectors, delimiter=', ', fmt='% s')
np.savetxt('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\ratings.csv', ratings, delimiter=', ', fmt='% s')

# Model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(max_words+1, embedding_vector_length, input_length=(reviews_train.shape[1])))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# loss1, acc1, mse1 =
print(model.evaluate(reviews_test, np.array(ratings_test)))
# print(f"Loss is {loss1},\nAccuracy is {acc1*100},\nMSE is {mse1}")
# history = model.fit(reviews_train, np.array(ratings_train), validation_split=0.2, epochs=5, batch_size=32, validation_data=(reviews_test, np.array(ratings_test)))

'''