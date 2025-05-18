 """
----------------------------------------------------------------------
🧠 Spam Detection Model Training Script (Unsupervised - Isolation Forest)
----------------------------------------------------------------------

This script prepares and trains an Isolation Forest model to detect spam 
messages using unsupervised anomaly detection. It extracts various 
features from text messages such as:
- Repetition ratio
- Compression ratio (via Huffman encoding)
- Exclamation and weird character ratios
- Presence of spam keywords
- Very long word detection

Only 'ham' (non-spam) messages are used for training the anomaly model.

✅ Output:
- Trained Isolation Forest model saved as 'iforest_model.pkl'

📄 Input:
- Dataset: spam.csv (expects columns 'v1' for label and 'v2' for message)

By Adel Muhammad Haiba | CS Student | Data Science & ML Enthusiast
"""


import pandas as pd
import numpy as np
import pickle
import string
from huffman import huffman_coding
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].dropna()
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

keywords = [
    'free', 'win', 'offer', 'click', 'buy', 'cheap', 'money', 'prize', 'urgent',
    'subscription', 'charged', 'confirm', 'reply', 'yes', 'no', 'claim', 'cash'
]

def repetition_ratio(text):
    words = text.split()
    if len(words) == 0:
        return 0
    unique_words = set(words)
    return 1 - (len(unique_words) / len(words))

def compression_ratio(text):
    if not text:
        return 0
    try:
        encoded, decoded, codes = huffman_coding(text)
        original_size = len(text) * 8
        compressed_size = len(encoded) if encoded else original_size
        return original_size / compressed_size if compressed_size != 0 else 0
    except:
        return 0

def exclamation_ratio(text):
    if not text:
        return 0
    return text.count('!') / len(text)

def weird_words_ratio(text):
    words = text.split()
    if not words:
        return 0
    weird_words = [w for w in words if len(w) <= 2 or w.lower() not in ENGLISH_STOP_WORDS]
    return len(weird_words) / len(words)

def weird_chars_ratio(text):
    if not text:
        return 0
    allowed_chars = set(string.ascii_letters + string.digits + ' ')
    weird_chars = [ch for ch in text if ch not in allowed_chars]
    return len(weird_chars) / len(text) if text else 0

def has_very_long_word(text, length_threshold=15):
    words = text.split()
    return any(len(w) > length_threshold for w in words)

def keyword_presence(text):
    text_lower = text.lower()
    return [1 if kw in text_lower else 0 for kw in keywords]

def uppercase_ratio(text):
    if not text:
        return 0
    uppercase_chars = sum(1 for ch in text if ch.isupper())
    return uppercase_chars / len(text) if text else 0

def special_symbols_ratio(text):
    if not text:
        return 0
    special_symbols = [ch for ch in text if ch in ['£', '$', '%', '&', '*', '#']]
    return len(special_symbols) / len(text) if text else 0

def raw_length(text):
    return len(text)

def extract_features(text):
    length = len(text.split())
    rep_ratio = repetition_ratio(text)
    comp_ratio = compression_ratio(text)
    excl_ratio = exclamation_ratio(text)
    weird_ratio = weird_words_ratio(text)
    weird_chars = weird_chars_ratio(text)
    very_long = 1 if has_very_long_word(text) else 0
    kw_presence = keyword_presence(text)
    upper_ratio = uppercase_ratio(text)
    special_ratio = special_symbols_ratio(text)
    text_length = raw_length(text)
    features = [length, rep_ratio, comp_ratio, excl_ratio, weird_ratio, weird_chars, very_long, upper_ratio, special_ratio, text_length] + kw_presence
    return features

X = np.array([extract_features(t) for t in df['v2']])
y = df['label'].values

model = IsolationForest(contamination=0.15, random_state=42)
model.fit(X)

with open('iforest_model_full.pkl', 'wb') as f:
    pickle.dump(model, f)

predictions = model.predict(X)
predicted_labels = [1 if pred == -1 else 0 for pred in predictions]  
accuracy = np.mean(predicted_labels == y)
print(f"Model trained on full dataset and saved successfully.")
print(f"Accuracy on the dataset: {accuracy:.4f}")


