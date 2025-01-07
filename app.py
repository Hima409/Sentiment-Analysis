from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('sentiment_model.h5')

# Load the tokenizer using pickle
with open("tokenizer.pkl", "rb") as f:
    t = pickle.load(f)

# Define the function to clean the review
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    text = ' '.join(words)
    return text

# Route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review_cleaned = clean(review)
    review_seq = t.texts_to_sequences([review_cleaned])
    review_pad = pad_sequences(review_seq, maxlen=500)
    prediction = model.predict(review_pad)
    result = "POSITIVE :)" if prediction > 0.5 else "NEGATIVE :("
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
