from pathlib import Path

from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# Initialize
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Load trained model and vectorizer
with (BASE_DIR / 'model.pkl').open('rb') as model_file:
    model = pickle.load(model_file)
with (BASE_DIR / 'vectorizer.pkl').open('rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Text preprocessing function
def transform_text(text):

    # lowercase
    text = text.lower()

    # tokenize
    words = wordpunct_tokenize(text)

    filtered_words = []

    for word in words:

        # keep only alphanumeric
        if word.isalnum():

            # remove stopwords and punctuation
            if word not in stop_words and word not in string.punctuation:

                # stemming
                filtered_words.append(ps.stem(word))

    return " ".join(filtered_words)


# Home page
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    message = request.form.get('message', '').strip()

    if not message:
        return render_template('index.html', error='Enter a message to analyze.'), 400

    # preprocess
    transformed = transform_text(message)

    # vectorize
    vector = vectorizer.transform([transformed])

    # prediction
    prediction = model.predict(vector)[0]

    # probability (safe method)
    try:
        probability = model.predict_proba(vector)[0][1]
        probability = round(probability * 100, 2)
    except (AttributeError, IndexError):
        probability = None

    if prediction == 1:
        result = "Spam Message ❌"
    else:
        result = "Not Spam ✅"

    return render_template(
        'index.html',
        prediction=result,
        probability=probability,
        message=message
    )


# Run server
if __name__ == "__main__":
    app.run(debug=True)
