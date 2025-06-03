from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Preprocessing function
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')]
    return ' '.join(tokens)

# Initialize Flask app
app = Flask(__name__)

# Routes
@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        cleaned_text = preprocess(news_text)
        vectorized_text = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        result = "REAL" if prediction == 1 else "FAKE"
        return render_template('index.html', prediction=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
