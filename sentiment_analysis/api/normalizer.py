from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Normalize the text data

# Strip white spaces function

def strip_whitespace(text):
    return text.strip()

# lower case function

def lower_case(text):
    return text.lower()

# remove punctuation function

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Tokenization function

def tokenize(text):
    return word_tokenize(text)

# Remove stopwords function

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [item for item in tokens if item not in stop_words]

# stemming function

def stem_text(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

# join words into a single string

def join_text(tokens):
    return ' '.join(tokens)


# join functions into a single function

def normalize_text(text):
    text = strip_whitespace(text)
    text = lower_case(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    text = join_text(text)
    return text




