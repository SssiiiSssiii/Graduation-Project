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
## remove_Stop_Words(tokens) function:
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
