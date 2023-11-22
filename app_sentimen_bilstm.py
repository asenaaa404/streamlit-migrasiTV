import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import subprocess

# Load model
model = load_model("sentiment_model.h5")

# Load Tokenizer
with open("sentiment_tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Fungsi Preprocessing
def preprocess_text(text):
    # Case Folding
    text = text.lower()

    # Menghapus username Twitter
    username_pattern = re.compile(r'@[\w_]+')
    text = re.sub(username_pattern, '', text)

    # Menghilangkan hashtag
    text = re.sub(r'#\w+', '', text)

    # Menghapus karakter-karakter tertentu
    text = re.sub(r'\b(n)([^a-zA-Z])', r'\2', text)
    text = re.sub(r'\n', '', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)
    text = re.sub('&quot;'," ", text)
    text = re.sub(r"\d+", " ", str(text))
    text = re.sub(r"\b[a-zA-Z]\b", "", str(text))
    text = re.sub(r"[^\w\s]", " ", str(text))
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = re.sub(r"\s+", " ", str(text))
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-z0-9]', ' ', str(text))
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s\s+', ' ', text)

    # Tokenisasi
    words = re.split(r'\W+', text.lower())

    # Slang Word
    kamus_slangword = pd.read_csv('slangwords.csv')
    kata_normalisasi_dict = {}

    for index, row in kamus_slangword.iterrows():
        if row[0] not in kata_normalisasi_dict:
            kata_normalisasi_dict[row[0]] = row[1]

    words = [kata_normalisasi_dict[term] if term in kata_normalisasi_dict else term for term in words]

    # Stopword Removal
    stopword = stopwords.words('indonesian')
    additional_stopwords = ["lama", "tidak", "platejohnny", "xad", "xba", "johnny", "johnny plate", "amp",
                            "xbb", "dvb", "xbc", "xbd", "xaa","xab", "bombunuhdiri", "lordranggameninggal", "andikceritanya",
                            "daftarslotonline","kegblgnunfaedh", "mhz"]
    stopword.extend(additional_stopwords)
    words = [word for word in words if word not in stopword]

    # Mengubah dari bentuk token menjadi bentuk kalimat kembali
    text = ' '.join(words)

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)

    return text

# Fungsi Prediksi
def predict_sentiment(sentence):
    # Tokenisasi dan padding pada kalimat yang ingin diprediksi
    sequences = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequences, maxlen=37, dtype='int32', value=0)

    # Melakukan prediksi sentimen untuk kalimat yang ingin diprediksi
    prediction = model.predict(padded)

    # Konversi hasil prediksi menjadi label sentimen
    label_sentimen = ["Negatif", "Positif", "Netral"]
    label_hasil = label_sentimen[np.argmax(prediction)]

    return label_hasil

# Streamlit App
st.title("Sentiment Analysis App")

# Input text
input_text = st.text_area("Enter a sentence:")

# Tombol untuk melakukan prediksi
if st.button("Predict Sentiment"):
    # Preprocess text
    preprocessed_text = preprocess_text(input_text)

    # Predict sentiment
    sentiment_prediction = predict_sentiment(preprocessed_text)

    # Tampilkan hasil prediksi
    st.success(f"Sentiment: {sentiment_prediction}")
