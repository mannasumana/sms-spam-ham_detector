from pathlib import Path
import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize


BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_resources():
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))

    with (BASE_DIR / 'model.pkl').open('rb') as model_file:
        model = pickle.load(model_file)
    with (BASE_DIR / 'vectorizer.pkl').open('rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer, stop_words


def transform_text(text, stop_words):
    stemmer = PorterStemmer()
    words = wordpunct_tokenize(text.lower())
    return ' '.join(
        stemmer.stem(word)
        for word in words
        if word.isalnum() and word not in stop_words and word not in string.punctuation
    )


st.set_page_config(page_title='AI SMS Spam Detector', page_icon='📩', layout='centered')
st.title('AI SMS Spam Detector')
st.caption('Analyze SMS messages with the trained machine-learning model.')

model, vectorizer, stop_words = load_resources()
message = st.text_area('Message', placeholder='Type a message to analyze...', height=160)

if st.button('Analyze message', type='primary', use_container_width=True):
    if not message.strip():
        st.warning('Enter a message to analyze.')
    else:
        vector = vectorizer.transform([transform_text(message, stop_words)])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1]

        if prediction == 1:
            st.error('Spam message')
        else:
            st.success('Not spam')

        st.metric('Spam probability', f'{probability * 100:.2f}%')
