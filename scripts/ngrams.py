import pandas as pd
from collections import Counter
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import string

def preprocess_text(text):
    """Lowercases, tokenizes, and removes punctuation from text."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return tokens

def get_common_ngrams(df, col, grams=1, top_n=20):
      """Performs frequency analysis on wordsin the given text columns."""
      all_text = " ".join(df[col].dropna().astype(str).values)
      tokens = preprocess_text(all_text)
      word_freq = Counter(ngrams(tokens, grams))
      print(f"Most common {grams}grams:")
      print(word_freq.most_common(top_n))

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('dataset.csv', sep='|')

    # Define text columns (adjust as needed)
    text_columns = ['description', 'solution']  # Change this to the actual text column(s) in your dataset

    # Run frequency analysis
    print(text_columns[0])
    for i in range(1, 15):
      get_common_ngrams(df, text_columns[0], i)
    print("############ BREAK ################")
  print(text_columns[1])
    for i in range(1, 15):
      get_common_ngrams(df, text_columns[1], i)
