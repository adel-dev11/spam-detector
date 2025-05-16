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

with open('iforest_model.pkl', 'rb') as f:
    model = pickle.load(f)

keywords = ['free', 'win', 'offer', 'click', 'buy', 'cheap', 'money', 'prize', 'urgent']

def repetition_ratio(text):
    words = text.split()
    if len(words) == 0:
        return 0
    unique_words = set(words)
    return 1 - (len(unique_words) / len(words))

def compression_ratio(text):
    if not text:
        return 0, 0, 0
    encoded, decoded, codes = huffman_coding(text)
    original_size = len(text) * 8  
    compressed_size = len(encoded) if encoded else original_size
    ratio = original_size / compressed_size if compressed_size != 0 else 0
    return original_size, compressed_size, ratio

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
    return len(weird_chars) / len(text)

def has_very_long_word(text, length_threshold=15):
    words = text.split()
    for w in words:
        if len(w) > length_threshold:
            return True
    return False

def keyword_presence(text):
    text_lower = text.lower()
    return [1 if kw in text_lower else 0 for kw in keywords]

def extract_features(text):
    length = len(text.split())
    rep_ratio = repetition_ratio(text)
    excl_ratio = exclamation_ratio(text)
    weird_ratio = weird_words_ratio(text)
    original_size, compressed_size, comp_ratio = compression_ratio(text)
    weird_chars = weird_chars_ratio(text)
    very_long = 1 if has_very_long_word(text) else 0
    kw_presence = keyword_presence(text)
    features = [length, rep_ratio, comp_ratio, excl_ratio, weird_ratio, weird_chars, very_long] + kw_presence
    return np.array(features).reshape(1, -1), original_size, compressed_size, comp_ratio

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        email_text = request.form['email']
        features, original_size, compressed_size, comp_ratio = extract_features(email_text)
        pred = model.predict(features)[0]  
        label = 'Spam' if pred == -1 else 'Ham'

        length = int(features[0][0])
        weird_ratio = float(features[0][4])
        weird_chars_val = float(features[0][5])
        very_long_val = int(features[0][6])

        if length > 50 and weird_ratio > 0.4:
            label = 'Spam'
        if weird_chars_val > 0.3:
            label = 'Spam'
        if very_long_val == 1:
            label = 'Spam'

        result = {
            'label': label,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': round(comp_ratio, 3),
            'length': length,
            'repetition_ratio': round(float(features[0][1]), 3),
            'exclamation_ratio': round(float(features[0][3]), 3),
            'weird_words_ratio': round(weird_ratio, 3),
            'weird_chars_ratio': round(weird_chars_val, 3),
            'very_long_word': bool(very_long_val),
            'keywords_found': [k for k, present in zip(keywords, features[0][7:]) if present == 1]
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
