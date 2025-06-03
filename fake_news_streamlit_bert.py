# ------------------------------
# 📊 Streamlit Fake News Detector with LSTM
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import class_weight

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

st.set_page_config(page_title="Fake News Detector (LSTM)", layout="centered")

# ------------------------------
# 🔧 Load and Clean Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("news.csv")
    df = df[['text', 'label']]
    df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})
    return df

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')]
    return ' '.join(tokens)

# ------------------------------
# 🧠 Prepare Model Data
# ------------------------------
@st.cache_resource
def prepare_model():
    df = load_data()
    df['clean_text'] = df['text'].apply(preprocess_text)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['clean_text'])
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    X = pad_sequences(sequences, maxlen=300)
    y = df['label'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=64, input_length=300))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.45))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    model.fit(X_train, y_train, epochs=7, batch_size=128, validation_data=(X_val, y_val), class_weight=class_weights)

    return model, tokenizer

model, tokenizer = prepare_model()

# ------------------------------
# 🌐 Streamlit App Interface
# ------------------------------

st.title("🧠 Fake News Detection using LSTM")

user_input = st.text_area("Paste a news article or headline:", height=200)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        with st.spinner("Analyzing with LSTM model..."):
            cleaned = preprocess_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=300)
            prediction = model.predict(padded)[0][0]

            if prediction > 0.45:
                st.success(f"✅ Prediction: REAL NEWS ({round(prediction*100, 2)}% confidence)")
            else:
                st.error(f"🚨 Prediction: FAKE NEWS ({round((1-prediction)*100, 2)}% confidence)")

# ------------------------------
# 🧪 Model Info
# ------------------------------
with st.expander("🔬 About the Model"):
    st.write("""
        - Model: LSTM from scratch
        - Layers: Embedding + LSTM + Dropout + Dense
        - Trained on Kaggle Fake/Real News Dataset
        - Outputs: Probability of REAL news
    """)
