import re
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    def __init__(self):
        # Initialize spaCy model for lemmatization
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Remove HTML tags using BeautifulSoup
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize the text into words
        tokens = word_tokenize(text)

        # Convert words to lowercase
        tokens = [word.lower() for word in tokens]

        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatize the tokens using spaCy
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        # Limit the number of tokens to 500
        tokens = tokens[:500]

        # Join the tokens back into a single string
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text
