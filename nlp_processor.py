"""
NLP Preprocessing module for Restaurant Review Analysis System.
Handles tokenization, stop word removal, and stemming.
"""

import re
import string
from typing import List

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class NLPProcessor:
    def __init__(self):
        self._download_nltk_data()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        # Add custom stop words relevant to restaurant reviews
        self.custom_stop_words = {
            'restaurant', 'food', 'place', 'order', 'ordered',
            'came', 'got', 'went', 'also', 'would', 'could'
        }
        self.stop_words.update(self.custom_stop_words)

    def _download_nltk_data(self):
        """Download required NLTK data."""
        nltk_packages = ['punkt', 'stopwords', 'punkt_tab']
        for package in nltk_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
            except LookupError:
                nltk.download(package, quiet=True)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)

    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """Remove punctuation from tokens."""
        return [token for token in tokens if token not in string.punctuation]

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove stop words from tokens."""
        return [token for token in tokens if token not in self.stop_words]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens."""
        return [self.stemmer.stem(token) for token in tokens]

    def preprocess(self, text: str, apply_stemming: bool = True) -> List[str]:
        """
        Full preprocessing pipeline.

        Args:
            text: Raw input text
            apply_stemming: Whether to apply stemming (default True)

        Returns:
            List of preprocessed tokens
        """
        # Clean text
        cleaned = self.clean_text(text)
        # Tokenize
        tokens = self.tokenize(cleaned)
        # Remove punctuation
        tokens = self.remove_punctuation(tokens)
        # Remove stop words
        tokens = self.remove_stop_words(tokens)
        # Apply stemming if requested
        if apply_stemming:
            tokens = self.stem_tokens(tokens)
        return tokens

    def preprocess_to_string(self, text: str, apply_stemming: bool = True) -> str:
        """
        Preprocess text and return as a single string.

        Args:
            text: Raw input text
            apply_stemming: Whether to apply stemming (default True)

        Returns:
            Preprocessed text as a single string
        """
        tokens = self.preprocess(text, apply_stemming)
        return ' '.join(tokens)


# Example usage and testing
if __name__ == "__main__":
    processor = NLPProcessor()

    sample_reviews = [
        "The pizza was absolutely delicious! Best I've ever had.",
        "Terrible service, waited 45 minutes for cold food.",
        "The pasta was okay, nothing special but not bad either.",
        "Amazing atmosphere and friendly staff. Will definitely come back!",
        "The steak was overcooked and the fries were soggy. Very disappointed."
    ]

    print("NLP Preprocessing Demo")
    print("=" * 50)

    for review in sample_reviews:
        print(f"\nOriginal: {review}")
        print(f"Processed: {processor.preprocess_to_string(review)}")
