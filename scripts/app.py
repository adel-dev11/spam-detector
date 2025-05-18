"""
simple Flask web application that detects spam messages
using an unsupervised machine learning model (Isolation Forest).

🧠 The model analyzes various features from the input text such as:
- Word repetition ratio
- Exclamation mark usage
- Presence of suspicious keywords
- Compression ratio using Huffman encoding
- Weird characters and very long words

🔍 The application returns a prediction (Spam or Ham) along with
detailed statistics about the message for transparency and analysis.

Author: Adel Muhammad Haiba
"""


from flask import Flask, render_template, request
import pickle
import numpy as np
import string
from huffman import huffman_coding
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

with open('iforest_model_full.pkl', 'rb') as f:
    model = pickle.load(f)

keywords = [
    'free', 'win', 'offer', 'click', 'buy', 'cheap', 'money', 'prize', 'urgent',
    'subscription', 'charged', 'confirm', 'reply', 'yes', 'no', 'mobile', 'claim', 'cash'
]

def repetition_ratio(text):
    words = text.split()
    if len(words) == 0:
        return 0
    unique_words = set(words)
    return 1 - (len(unique_words) / len(words))

def compression_ratio(text):
    if not text:
        return 0, 0, 0
    try:
        encoded, decoded, codes = huffman_coding(text)
        original_size = len(text) * 8
        compressed_size = len(encoded) if encoded else original_size
        ratio = original_size / compressed_size if compressed_size != 0 else 0
        return ratio, original_size, compressed_size
    except:
        return 0, 0, 0

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
    comp_ratio, original_size, compressed_size = compression_ratio(text)
    excl_ratio = exclamation_ratio(text)
    weird_ratio = weird_words_ratio(text)
    weird_chars = weird_chars_ratio(text)
    very_long = 1 if has_very_long_word(text) else 0
    kw_presence = keyword_presence(text)
    upper_ratio = uppercase_ratio(text)
    special_ratio = special_symbols_ratio(text)
    text_length = raw_length(text)
    features = [length, rep_ratio, comp_ratio, excl_ratio, weird_ratio, weird_chars, very_long, upper_ratio, special_ratio, text_length] + kw_presence
    return np.array(features).reshape(1, -1), original_size, compressed_size, comp_ratio

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        email_text = request.form['email']
        
        if not email_text.strip():
            return render_template('index.html', result={'error': 'Please enter a valid email text.'})

        features, original_size, compressed_size, comp_ratio = extract_features(email_text)
        
        if comp_ratio > 2.7:
            label = 'Spam'
        else:
            pred = model.predict(features)[0]
            label = 'Spam' if pred == -1 else 'Ham'
        
        result = {
            'label': label,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': round(comp_ratio, 3),
            
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
