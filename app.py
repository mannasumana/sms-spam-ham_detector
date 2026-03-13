from flask import Flask, render_template, request
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    words = word_tokenize(text)

    y = []
    for word in words:
        if word.isalnum():
            if word not in stopwords.words('english') and word not in string.punctuation:
                y.append(ps.stem(word))

    return " ".join(y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    # preprocess text
    transformed = transform_text(message)

    # vectorize
    vector = vectorizer.transform([transformed])

    # prediction
    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0][1]

    if prediction == 1:
        result = "Spam Message ❌"
    else:
        result = "Not Spam ✅"

    return render_template(
        'index.html',
        prediction=result,
        probability=round(probability*100,2),
        message=message
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)