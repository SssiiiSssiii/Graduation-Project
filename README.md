# Graduation-Project
sentiment analysis            
Drive : https://drive.google.com/drive/u/2/folders/1sHGVQ6xxHyAC77hUPb5y3AxUj0buW4nf


## Code
```c
import numpy as np
import pandas as pd
from nltk import ISRIStemmer
from pyarabic import araby
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings("ignore")
from stop_words import ArabicStopWords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, metrics


# def word_extraction():

# Taking a File
def tokenize(sample):
    tokens = araby.tokenize(sample)
    return tokens

# Taking a List
def remove_Stop_Words(tokens):
    stop_Words = ArabicStopWords.getStopWords("")
    filtered = []
    for token in tokens:
        if token not in stop_Words:
            filtered.append(token)
    return filtered


def print_Stop_Words(tokens):
    stop_Words =  ArabicStopWords.getStopWords("")
    filtered = []
    for token in tokens:
        if token in stop_Words:
            filtered.append(token)
    return filtered

# Taking a List
def stemming(filtered_Tokens):
    stemmer = ISRIStemmer()
    stemmed = []

    for token in filtered_Tokens:
        stemmed.append(stemmer.stem(token))

    return stemmed

# Taking a word
def normalize(token):
    token = re.sub("[إأآا]", "ا", token)
    token = re.sub("ى", "ي", token)
    token = re.sub("ة", "ه", token)
    token = re.sub("[\W\d]", "", token)
    token = araby.strip_diacritics(token)
    token = araby.strip_tatweel(token)
    return token

# Reading Dataset
cols = ['Class', 'Text']
sample = pd.read_csv('Data_Set.txt', sep='\t', header=None, names=cols)


# Preprocessing / Cleaning Dataset
# Tokenization
for row in range(len(sample)):
    sample['Text'][row] = tokenize(sample['Text'][row])


# Normalization
for row in range(len(sample)):
    for token in range(len(sample['Text'][row])):
         sample['Text'][row][token] = normalize(sample['Text'][row][token])

# Removing Stop Words
for row in range(len(sample)):
    sample['Text'][row] = remove_Stop_Words(sample['Text'][row])

# Stemming
for row in range(len(sample)):
    sample['Text'][row] = stemming(sample['Text'][row])

#Features Extraction from rows text with TFIDF
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='word', ngram_range=(1, 1), max_features =10000)
tfidf = tfidf_vectorizer.fit_transform(sample['Text'].astype('str'))

unigramdataGet = tfidf.toarray()
vocab = tfidf_vectorizer.get_feature_names_out()
features = pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
features[features>0] = 1

# print(features.iloc[1:5,1:5])

# Training and Testing with Machine Learning Algorithms
pro = preprocessing.LabelEncoder()
encpro = pro.fit_transform(sample['Class'])
sample['Class'] = encpro

# Training
y = sample['Class']
X = features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=333)
nb = MultinomialNB()
nb = nb.fit(X_train, y_train)

# Testing
y_pred = nb.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
# disp.plot()
# plt.show()

# print('Accuracy = ', metrics.accuracy_score(y_test,  y_pred))
# print('Precision', metrics.precision_score(y_test, y_pred))

LR = LogisticRegression(penalty = 'l2', C = 1)
LR = LR.fit(X_train , y_train)

# Prediction on sample text
Sample_Text = 'لم ذهبت إلى المدرسة '

tokens = tokenize(Sample_Text)
print("Tokens is ----> ", tokens)

# Normalizing
for row in range(len(tokens)):
    for token in range(len(tokens)):
         tokens[token] = normalize(tokens[token])

print("After Normalization ----> ", tokens)

# Removing stop_words
tokens = remove_Stop_Words(tokens)

print("After Remove Stop words ----> ", tokens)

# Stemming
tokens = stemming(tokens)
print("After Stemming ----> ", tokens)

tv = [str(tokens)]

x = tfidf_vectorizer.transform(tv)
pred = LR.predict(x)
pred = pro.inverse_transform(pred)
print(''.join(pred))



```

