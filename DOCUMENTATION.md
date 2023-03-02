
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

## Feature Extraction
```C
# Load the dataset
sample = pd.read_csv('100k.csv')

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
* Loads a CSV file named '100k.csv' into a Pandas DataFrame called sample.
* Creates a TfidfVectorizer object called tfidf_vectorizer and fits it to the text data in the Text column of the sample DataFrame. The TfidfVectorizer object applies sublinear scaling to use smaller weights for higher frequency words, strips accents from words, uses words as the features, uses unigrams (single words) as the features, and limits the number of features to the top 500,000 by frequency.
* Computes the TF-IDF matrix for the text data using the tfidf_vectorizer object and stores it in a sparse matrix called tfidf.
* Gets the feature names (vocabulary) of the TF-IDF matrix using the get_feature_names() method of the tfidf_vectorizer object and stores them in a list called vocab.
* Converts the tfidf sparse matrix to a dense NumPy array called unigram_data.
* Converts the dense array to a Pandas DataFrame called features and rounds the values to one decimal place.
* Binarizes the features DataFrame by setting non-zero values to 1.
* Saves the fitted TfidfVectorizer object to a file named '100k.pkl' using the pickle.dump() function.
* Saves the computed TF-IDF matrix to a file named '100k2.pkl' using the pickle.dump() function.

