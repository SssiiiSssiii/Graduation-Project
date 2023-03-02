import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the dataset
sample = pd.read_csv('DataSet.csv')

# Create a TfidfVectorizer object and fit to data
tfidf_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 1),
    max_features=500000
)
tfidf = tfidf_vectorizer.fit_transform(sample['Text'].astype(str))

# Get feature names (vocabulary)
vocab = tfidf_vectorizer.get_feature_names()

# Convert tfidf matrix to DataFrame
unigram_data = tfidf.toarray()
features = pd.DataFrame(np.round(unigram_data, 1), columns=vocab)
features[features > 0] = 1

# Save the fitted TfidfVectorizer and computed TF-IDF matrix to disk
with open('DataSet.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('DataSet2.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
