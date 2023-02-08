# Warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import re

import numpy as np
import stopwords as stopwords
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Imports
import pandas as pd
import gzip
import json
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download('wordnet')
# nltk.download('stopwords')
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer


#from nltk.stem import PorterStemmer
# import numpy as np
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# import os
# import datetime
# import tensorflow.keras
# from tesnorflow.keras.model import Sequential
# from nlkt.tokenize import word_tokenize

# warnings.filterwarnings('ignore')
################################################## INITIAL PREPARATION #################################################
# Reading in file & converting to dataframe
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path):
    i=0
    df={}
    for d in parse(path):
        df[i]=d
        i+=1
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF('C:\WGU\D213 Advanced Data Analytics\Task 2\All_Beauty_5.json.gz')

# Analyzing dataframe
print(df.shape)
print(df.columns)
for col in df:
    print(df[col].head(10))

# Renaming "overall" to "rating" to provide a clearer understanding
df = df.rename(columns={'overall' : 'rating'})

################################################## HANDLING NULLS ######################################################
# Finding the total number of null values in each column
def sumNulls(df):
    i=0
    print("The number of null values in each column are:")
    for col in df:
        print(list(df.columns.values)[i])
        print(sum(pd.isnull(df[col])))
        i+=1

sumNulls(df)

# New dataframe of only Rating and Review Text to perform analysis -> dfRR for Rating and Review
dfRR = df[['rating', 'reviewText']]
print(dfRR.head(10))

sumNulls((dfRR))

# Dropping five rows with null Review Text
dfRR = dfRR.dropna()
sumNulls(dfRR)
print(dfRR.shape)

################################################## CONVERT RATING TO POSITIVE OR NEGATIVE ##############################
print(dfRR['rating'].value_counts())

# Removing "neutral" sentiments & renaming dataframe as dfPN for Postivive/Negative
dfPN = dfRR[dfRR['rating'] != 3]
print(dfPN['rating'].value_counts())

# Assigning negative scores to 0
dfPN.loc[dfPN['rating'] == 1, 'rating'] = 0
dfPN.loc[dfPN['rating'] == 2, 'rating'] = 0
print(dfPN['rating'].value_counts())

# Assigning positive scores to 1
dfPN.loc[dfPN['rating'] == 4, 'rating'] = 1
dfPN.loc[dfPN['rating'] == 5, 'rating'] = 1
print(dfPN['rating'].value_counts())
print(dfPN.shape)
################################################## REVIEW TEXT HANDLING ################################################
# Convert to lowercase and remove punctuation
def isEnglish(dataframe):
    notAsciiCount = 0
    for x in dataframe:
        if not x.isascii():
            notAsciiCount += 1
    return notAsciiCount

print("Count of reviews with non-English characters: ", isEnglish(dfPN['reviewText']), '\n')

dfPN['reviewText'] = dfPN['reviewText'].apply(lambda x: ''.join(char for char in x if ord(char) < 128))
dfPN['reviewText'] = dfPN['reviewText'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
dfPN['reviewText'] = dfPN['reviewText'].str.lower()


cleaned = RegexpTokenizer('\w+')
dfPN['tok'] = dfPN['reviewText'].apply(cleaned.tokenize)
print(dfPN.head(10))


# stopwords = set(stopwords.words('english'))
#
# def remove_stopwords(text):
#     output = [i for i in text if i not in stopwords]
#     return output
#
# dfPN['no_stop'] = dfPN['tok_res'].apply(lambda x: remove_stopwords(x))
# print("Stopwords Removed:")
# print(dfPN['no_stop'])

# wordnet_lemmatizer = WordNetLemmatizer()
#
# def lemmatizer(text):
#     lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
#     return lemm_text
#
# dfPN['lem'] = dfPN['no_stop'].apply(lambda x: lemmatizer(x))
# print("Stopwords Removed & Lemmatized")
# print(dfPN['lem'])

# Wordcloud stuff - keep for later

# tokenized = Tokenizer(num_words=None)
# tokenized.fit_on_texts(dfPN['no_stop'])
# X_train = tokenized.texts_to_sequences(dfPN['no_stop'])
# print("How many words: ", len(X_train))
# print("Vocabulary: ", len(tokenized.word_index) + 1)
# print("Max Length: ", max(len(word) for word in X_train))

################################################## REVIEW TEXT WORDCLOUDS ##############################################

# # Wordcloud for all reviews
# def lem_string(list):
#     string = ' '.join(list)
#     return string
#
# dfPN['lem_string'] = dfPN['lem'].apply(lambda x: lem_string(x))
# print(dfPN['lem_string'])
#
# text = "".join(x for x in dfPN['lem_string'])
# wordcloud = WordCloud(stopwords=stopwords).generate(text)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.savefig("wordcloud1.png")
# plt.show()

# # Wordcloud for positive reviews
# dfPos = dfPN[dfPN['rating'] == 1]
# positive = "".join(x for x in dfPos['lem'])
# wordcloudPos = WordCloud(stopwords=stopwords).generate(positive)
# plt.imshow(wordcloudPos, interpolation='bilinear')
# plt.axis("off")
# plt.savefig("wordcloudPos.png")
# plt.show()
#
# # Wordcloud for positive reviews
# dfNeg = dfPN[dfPN['rating'] == 0]
# negative = "".join(x for x in dfNeg['lem'])
# wordcloudNeg = WordCloud(stopwords=stopwords).generate(negative)
# plt.imshow(wordcloudNeg, interpolation='bilinear')
# plt.axis("off")
# plt.savefig("wordcloudNeg.png")
# plt.show()

################################################## SETTING PARAMETERS ##################################################
all_words = []
for x in dfPN['tok']:
    for i in x:
        all_words.append(i)

unique_words = set(all_words)
max_words = len(unique_words)
print("Count of unique words:", max_words)
# End borrowed code

longest_review = 0
for x in dfPN['tok']:
    if len(x) > longest_review:
        longest_review = len(x)

print("The review with the most words has", longest_review, "words.\n")

################################################## VECTORIZING & PADDING ###############################################
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(dfPN['tok'])
seq = tokenizer.texts_to_sequences(dfPN['tok'])
print("Embedded sentences:")
print(seq[0:10], '\n')
review_vectors = pad_sequences(seq, padding='post', maxlen=longest_review)
print("Padded sentences:")
print(review_vectors[0:10], '\n')

################################################## TRAIN/TEST SPLIT ####################################################
ratings = list(dfPN['rating'])

reviews_train, reviews_test, ratings_train, ratings_test = train_test_split(review_vectors, ratings, test_size=0.2,
                                                                            random_state=22)

print("Reviews Train:\n", reviews_train[0:10])
print("Reviews Test:\n", reviews_test[0:10])
print("Ratings Train:\n", ratings_train[0:10])
print("Ratings Test:\n", ratings_test[0:10])

# np.savetxt('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\review_vextors_2.csv', review_vectors, delimiter=', ', fmt='% s')
# np.savetxt('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\ratings_2.csv', ratings, delimiter=', ', fmt='% s')


################################################## MODELING & PREDICTING ###############################################
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_vector_length, input_length=longest_review))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(units=128))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# loss1, acc1, mse1 =
print(model.evaluate(reviews_test, np.array(ratings_test)))
# history = model.fit(reviews_train, np.array(ratings_train), validation_split=0.2, epochs=5, batch_size=32,
#                     validation_data=(reviews_test, np.array(ratings_test)))
