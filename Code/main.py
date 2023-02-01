import pandas as pd
import re

from nltk.corpus import stopwords

colnames = ['review', 'rating']
yelp = pd.read_csv('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\yelp_labelled.txt',
                   sep='\t', header=None, encoding='latin-1', names=colnames)
imdb = pd.read_csv('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\imdb_labelled.txt',
                   sep='\t', header=None, names=colnames)
amzn = pd.read_csv('C:\\WGU\\D213 Advanced Data Analytics\\Task 2\\amazon_cells_labelled.txt',
                   sep='\t', header=None, names=colnames)

df = pd.concat((yelp, imdb, amzn), ignore_index=True)
pd.set_option('max_colwidth', 30)
pd.set_option('display.max_columns', 10)

print(df.info())
print(df.head(10), '\n')


df['word_count'] = [len(x.split()) for x in df['review']]
df['char_count'] = df['review'].apply(len)
print(df)
print(df['rating'].value_counts(), '\n')
print("Max word count: ", max(len(x.split()) for x in df['review']), '\n')

# Review Cleanup
df['review'] = df['review'].astype(str).str.lower()
df['review_clean'] = df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
print(df['review_clean'], '\n')

stopwords = set(stopwords.words('english'))

df['no_stop'] = df['review_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
print(df['no_stop'].head(10), '\n')

max_words = max(len(x.split()) for x in df['no_stop'])
print("Max word count: ", max_words)


