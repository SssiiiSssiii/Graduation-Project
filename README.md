![alt text](PicsArt_03-27-03.24.51.png)


# Arabic Sentiment Analysis
Is a process of extracting the sentiment or emotion expressed in a given text in the Arabic language. This code provides a machine learning-based approach to classify Arabic text into positive, negative, or neutral sentiment categories. It uses natural language processing (NLP) techniques such as tokenization, stemming, and stop-word removal to preprocess the Arabic text data. The model is trained on a labeled dataset of Arabic text and evaluated using various performance metrics such as accuracy, precision, recall, and F1-score. The code is implemented in Python and requires the installation of several libraries such as NLTK, PyArabic, and Scikit-learn. This code can be used for various applications, such as sentiment analysis of customer reviews, social media posts, and news articles in the Arabic language.

# Dependencies
This code requires the following libraries:   
* pickle
* numpy
* pandas
* nltk (specifically the ISRIStemmer module)
* pyarabic
* sklearn (specifically the LogisticRegression, MultinomialNB, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, train_test_split, TfidfVectorizer, CountVectorizer, preprocessing, and metrics modules)
* stop_words (specifically the ArabicStopWords module is our built-in module)
* warnings (to suppress warnings)
* re (for regular expressions)
# Installation
To install the required libraries, run:
```c
pip install -r requirements.txt
```
