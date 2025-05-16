# 🛡️ Spam Email Detector Using Isolation Forest & Flask

A lightweight web app that uses **unsupervised machine learning** (Isolation Forest) to detect spam emails based on **text features and compression**.

The model learns to detect **outlier messages** (spam) by training only on normal messages (ham).  
It also uses **Huffman encoding compression** to help spot unnatural text patterns.

---

## 🚀 Features

- Unsupervised Spam Detection (Isolation Forest)
- Huffman Compression-based Feature
- Handles gibberish & suspicious long texts
- Flask Web Interface
- Highlighted keyword detection (`free`, `win`, `buy`, etc.)
- Supports Arabic/English UI

---

## 🧠 How It Works

The model extracts features such as:
- Number of words
- Repetition ratio
- Compression ratio
- Exclamation mark frequency
- Ratio of unknown or weird words
- Presence of spammy keywords
- Ratio of weird characters
- Detection of overly long words

These features are passed to an `IsolationForest` model trained only on clean (ham) messages.

---

## 📁 Files Structure

```bash
SpamDetector/
├── app.py                    # Flask web application
├── train_model.py            # Script to train and save the Isolation Forest model
├── huffman.py                # Huffman encoding algorithm used in compression ratio
├── spam.csv                  # Dataset used for training (must include 'ham'/'spam' labels)
├── iforest_model.pkl         # Trained Isolation Forest model
├── README.md                 # Project documentation (this file)
│
├── templates/
│   └── index.html            # HTML template for the web app
│
└── static/
    └── styles.css            # Optional CSS file for styling
---

## 🙌 Acknowledgements

- Dataset from [UCI Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Huffman compression logic adapted for educational use
---

## 📄 License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute it with proper attribution.

## 👨‍💻 About Me

**Name**: Adel Muhammad Haiba  
**GitHub**: adel-dev11 (https://github.com/adel-dev11)
**Email**: adeldevj@gmail.com  
**Role**: Data Science & Machine Learning  
**Tech Stack**: Python, Flask, Scikit-learn, HTML, CSS
