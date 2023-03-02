import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings("ignore")

# Load the TfidfVectorizer object and TF-IDF matrix from disk
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Get feature names (vocabulary) and convert TF-IDF matrix to DataFrame
vocab = tfidf_vectorizer.get_feature_names()
unigram_data = tfidf.toarray()
features = pd.DataFrame(np.round(unigram_data, 1), columns=vocab)
features[features > 0] = 1

try:
    # Load trained models and label encoder from disk
    with open('nb_model.pkl', 'rb') as f:
        nb = pickle.load(f)
    with open('lr_model.pkl', 'rb') as f:
        LR = pickle.load(f)
    with open('pro_encoder.pkl', 'rb') as f:
        pro = pickle.load(f)
    print('Trained models loaded from disk')
except FileNotFoundError:
    # If trained models don't exist, train and save them

    pro = preprocessing.LabelEncoder()

    preprocessed_data = pd.read_csv('modified_data.csv')
    preprocessed_features = features

    # fit the encoder with the target variable data
    pro.fit(preprocessed_data['Class'])

    # Transform the target variable using the pre-trained encoder
    encpro = pro.transform(preprocessed_data['Class'])
    y = encpro

    # Transform the input features using the preprocessed features
    X = preprocessed_features

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=333)

    # Train a Multinomial Naive Bayes model and a logistic regression model
    nb = MultinomialNB()
    nb = nb.fit(X_train, y_train)

    LR = LogisticRegression(penalty='l2', C=1)
    LR = LR.fit(X_train, y_train)

    # Save the trained models and label encoder to disk
    with open('nb_model.pkl', 'wb') as f:
        pickle.dump(nb, f)
    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(LR, f)
    with open('pro_encoder.pkl', 'wb') as f:
        pickle.dump(pro, f)

    print('Trained models saved to disk')
