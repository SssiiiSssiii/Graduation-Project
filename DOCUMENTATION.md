
## tokenize(sample) function:
* Tokenizes an Arabic text into a list of tokens. 
* This function uses the araby library, which is a Python package for working with Arabic text.

### Parameters:
* Sample: A string containing the Arabic text to be tokenized.        
### Returns:
* A list of tokens, where each token is a string representing a word in the input text.         
### Example Usage:
```c
import araby

def tokenize(sample):
    tokens = araby.tokenize(sample)
    return tokens

text = "مرحبا بالعالم"
tokens = araby.tokenize(text)
print(tokens)
# Output: ['مرحبا', 'بالعالم']
```

## normalize(token) function:
* Is used to perform normalization on Arabic text.
* Normalization is the process of transforming text into a standard form, to remove any inconsistencies in the text. 
* In this function, normalization is performed by replacing `similar Arabic letters with a single letter`, `removing diacritics`, and `removing tatweel` (elongation) characters.

### Parameters:
* token: a string representing an Arabic word.  
### Returns:
* A normalized string representing the input token after applying the following transformations:         
    - Replacing Arabic letters 'إ', 'أ', 'آ', and 'ا' with 'ا'.       
    - Replacing Arabic letter 'ى' with 'ي'.           
    - Replacing Arabic letter 'ة' with 'ه'.              
    - Removing all non-alphabetic characters and digits using regular expressions.             
    - Removing all diacritics from the Arabic text.            
    - Removing all tatweel (elongation) characters.               
### Example Usage:
```c
from pyarabic import araby
import re

def normalize(token):
    token = re.sub("[إأآا]", "ا", token)
    token = re.sub("ى", "ي", token)
    token = re.sub("ة", "ه", token)
    token = re.sub("[\W\d]", "", token)
    token = araby.strip_diacritics(token)
    token = araby.strip_tatweel(token)
    return token
    
print(normalize("وعُدتُ مِنَ المعاركِ لستُ أدري علامَ أضعتُ عمري فـــي النِّزالِ ##1"))

# Output: وعدت من المعارك لست ادري علام أضعت عمري في النزال
```
## remove_Stop_Words(tokens) function:
* Removes stop words from a list of tokens using `ArabicStopWords` class(our built-in class).

### Parameters:
* tokens (list of str): A list of Arabic text tokens.

### Returns:
* filtered (list of str): A list of tokens without stop words.

Example Usage:
```c
def remove_Stop_Words(tokens):
    stop_words = ArabicStopWords()
    filtered = [token for token in tokens if token not in stop_words.get_stop_words()]
    return filtered 
    
tokens = ['في', 'المدرسة', 'نحن', 'نتعلم', 'الرياضيات']
filtered = remove_Stop_Words(tokens)
print(filtered)
# Output:['المدرسة', 'نتعلم', 'الرياضيات']
```

## stemming(filtered_tokens) function:
* This function takes a list of filtered tokens, creates an instance of the ISRIStemmer class (an Arabic stemmer), and uses it to stem each token in the list. 

### Parameters:
* filtered_tokens (list): A list of Arabic words to be stemmed.     
### Returns:
* list: A list of stemmed Arabic words.     
### Example Usage:
```c
from nltk import ISRIStemmer

def stemming(filtered_tokens):
    stemmer = ISRIStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens
    
filtered_tokens = ['يشرب', 'محمد']
print(stemming(filtered_tokens))

# Output: ['شرب', 'حمد']
```

## Feature Extraction using TF-IDF
```C
# Load the dataset
sample = pd.read_csv('DS.csv')

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
```
* Loads a CSV file named `'DS.csv'` into a Pandas DataFrame called sample.
* Creates a TfidfVectorizer object called `tfidf_vectorizer` and fits it to the text data in the Text column of the sample DataFrame. The TfidfVectorizer object applies sublinear scaling to use smaller weights for higher frequency words, strips accents from words, uses words as the features, uses unigrams (single words) as the features, and limits the number of features to the top `500,000` by frequency.
* Computes the `TF-IDF matrix` for the text data using the `tfidf_vectorizer` object and stores it in a sparse matrix called `tfidf`.
* Gets the feature names (vocabulary) of the TF-IDF matrix using the get_feature_names() method of the tfidf_vectorizer object and stores them in a list called `vocab`.
* Converts the tfidf sparse matrix to a dense NumPy array called `unigram_data`.
* Converts the dense array to a Pandas DataFrame called features and rounds the values to one decimal place.
* Binarizes the features DataFrame by setting non-zero values to 1.
* Saves the fitted TfidfVectorizer object to a file named `'DS.pkl'` using the pickle.dump() function.
* Saves the computed TF-IDF matrix to a file named `'DS2.pkl'` using the pickle.dump() function.
## Training using naive bayes

```c
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

```
* Load the `tfidf_vectorizer.pkl` and `tfidf_matrix.pkl` files into their respective objects using pickle.
* Extract the feature names (vocabulary) using the get_feature_names() method on tfidf_vectorizer object.
* Transform the `tfidf matrix` into a pandas DataFrame with binary values using np.round and pd.DataFrame.
* Load trained models and pre-trained preprocessing.LabelEncoder object, if they exist.
* If the trained models do not exist, pre-process the input features and train `MultinomialNB`
* Save the trained model and preprocessing.LabelEncoder object to disk using pickle and print a message to confirm the save operation.

## Prediction
```c
# Prediction on sample text
sample_text = input('Enter Text \n')

tokens = tokenize(sample_text)
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
pred = nb.predict(x)
prob = nb.predict_proba(x)

# Check if the predicted probability for both positive and negative classes is below a threshold
if prob[0][0] < 0.6 and prob[0][1] < 0.6:
    pred = ['neutral']
else:
    pred = pro.inverse_transform(pred)

# Print the predicted class and probabilities for each class
classes = nb.classes_
for i, p in enumerate(prob[0]):
    print(f"{classes[i]}: {p:.4f}")

print('Predicted class:', ''.join(pred))

```
This code predicts the sentiment class of a given input text using a Naive Bayes model that was trained on preprocessed text data. The prediction process involves the following steps:

1. The input text is first tokenized using the `tokenize()` function.
2. The resulting tokens are normalized using the `normalize()` function.
3. Stop words are removed from the normalized tokens using the `remove_Stop_Words()` function.
4. The remaining tokens are stemmed using the `stemming()` function.
5. The preprocessed tokens are then transformed using the pre-trained `TF-IDF vectorizer`, `tfidf_vectorizer`.
6. The transformed tokens are passed to the trained `Naive Bayes` model, nb, to predict the sentiment class of the input text.
7. The predicted class and probabilities for each class are printed to the console. If the predicted probability for both `positive and `negative classes is below a threshold of 0.6, the predicted class is set to `neutral`.
8. The final predicted class is printed to the console.
