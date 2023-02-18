# Graduation-Project
sentiment analysis            
Drive : https://drive.google.com/drive/u/2/folders/1sHGVQ6xxHyAC77hUPb5y3AxUj0buW4nf


## tokenize(sample) function:
* Tokenizes an Arabic text into a list of tokens. This function uses the araby library, which is a Python package for working with Arabic text.

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
## remove_Stop_Words function:
* Removes stop words from a list of tokens using ArabicStopWords.

### Parameters:
* tokens (list of str): A list of Arabic text tokens.

### Returns:
* filtered (list of str): A list of tokens with stop words removed.

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
* The remove_Stop_Words() function takes a list of tokens as input and removes Arabic stop words from the list. The function uses the ArabicStopWords() class from the arabicstopwords package to get a list of Arabic stop words, and then filters the input list to remove any tokens that appear in the stop words list.

* The function returns a new list containing only the tokens that were not stop words. If the input list contains no stop words, the function will simply return a copy of the original list.
```c

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

The code imports various libraries, including numpy, pandas, nltk, pyarabic, and sklearn. These libraries provide various functions for data analysis, machine learning, and natural language processing.

It then defines several functions to process the input text data. tokenize tokenizes the text into individual words, remove_Stop_Words removes stop words, stemming stems the words using ISRI stemmer, and normalize normalizes the text by removing diacritics and special characters.

Next, the code reads in a dataset with columns 'Class' and 'Text' using pd.read_csv and stores it in the sample variable. The code then performs several pre-processing steps on the dataset: tokenization, normalization, removing stop words, and stemming.

The code then uses the TfidfVectorizer from sklearn to extract features from the pre-processed text data, converting it into a numerical form that can be used for training machine learning algorithms.

The code then uses LabelEncoder from sklearn.preprocessing to encode the class labels, transforming the class labels from string values to numerical values.

The code then uses the MultinomialNB and LogisticRegression algorithms from sklearn to train and test the dataset, splitting it into training and testing datasets using train_test_split. The code then makes predictions on a sample text using the trained models.

Finally, the code calculates and displays the performance metrics of the models, such as confusion matrix, f1-score, and precision score.
