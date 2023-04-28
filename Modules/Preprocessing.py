import numpy as np
import pandas as pd
from nltk import ISRIStemmer
from pyarabic import araby
import warnings
warnings.filterwarnings("ignore")
from stop_words import ArabicStopWords
import re

def tokenize(sample):
    tokens = araby.tokenize(sample)
    return tokens

# Taking a List
def remove_Stop_Words(tokens):
    stop_words = ArabicStopWords()
    filtered = [token for token in tokens if token not in stop_words.get_stop_words()]
    return filtered


def print_Stop_Words(tokens):
    stop_words = ArabicStopWords()
    filtered = [token for token in tokens if token in stop_words.get_stop_words()]
    return filtered

# Taking a List
def stemming(filtered_tokens):
    stemmer = ISRIStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

# Taking a word
def normalize(token):
    token = re.sub("[إأآا]", "ا", token)
    token = re.sub("ى", "ي", token)
    token = re.sub("ة", "ه", token)
    token = re.sub("[\W\d]", "", token)
    token = araby.strip_diacritics(token)
    token = araby.strip_tatweel(token)
    return token

import pandas as pd

# Reading Dataset
file_path = 'DataSet.txt'
cols = ['Class', 'Text']
sample = pd.DataFrame(columns=cols)

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            line_parts = line.strip().split('\t')
            class_label = line_parts[0]
            text = line_parts[1]
            new_row = pd.DataFrame({'Class': class_label, 'Text': text}, index=[0])
            sample = pd.concat([sample, new_row], ignore_index=True)
        except:
            pass


# Preprocessing / Cleaning Dataset

# Tokenization
for row in range(len(sample)):
    sample['Text'][row] = tokenize(sample['Text'][row])

# Normalization
for row in range(len(sample)):
    sample['Text'][row] = [normalize(token) for token in sample['Text'][row]]

# Removing Stop Words
for row in range(len(sample)):
    sample['Text'][row] = remove_Stop_Words(sample['Text'][row])

# Stemming
for row in range(len(sample)):
    print("The ain't believe in us 2")
    sample['Text'][row] = stemming(sample['Text'][row])

# Removing null values
for row in range(len(sample)):
    sample['Text'][row] = [text for text in sample['Text'][row] if text != '']
    if not sample['Text'][row]:  # check if the list is empty
        sample.drop(row, inplace=True)


sample.to_csv('DataSet.csv', index=False)


