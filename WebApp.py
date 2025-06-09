import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Streamlit app title
st.title("AI vs Human Text Classifier")

# Load the trained model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('text_classification_lstm.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Input text area
user_input = st.text_area("Enter text to classify:", height=200)

# Button to trigger classification
if st.button("Classify Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Preprocess the input text
        max_len = 100  # Must match training max_len
        sequences = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequences, maxlen=max_len)
        
        # Make prediction
        prediction = model.predict(padded, verbose=0)[0][0]
        
        # Display result
        label = "AI-generated" if prediction > 0.5 else "Human-written"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        st.write(f"**Prediction**: {label}")
        st.write(f"**Confidence**: {confidence * 100:.2f}%")