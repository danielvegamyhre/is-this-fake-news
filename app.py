import re
import pickle
import numpy as np
import pandas as pd
import torch
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request, jsonify

# import MLP module definition
from models import MLP

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

# load saved model parameters and vectorizers
model = CustomUnpickler(open('data/multi-layer-perceptron-parameters.pkl', 'rb')).load()
text_vectorizer = CustomUnpickler(open('data/text_vectorizer.pkl','rb')).load()


def preprocess(df):
    """
    Preprocess user input in the same way we preprocessed the training data.

    1. Remove non-alphabetic characters, convert to lowercase
    2. Tokenize (word_tokenizer from nltk)
    3. Lemmatize (WordNetLemmatizer)
    4. Vectorize (CountVectorizer)

    Use the same CountVectorizers from training in order to extract
    the same features and have the same output dimensions.
    """
    lemmatizer = WordNetLemmatizer()

    text_processed = []
    for text in df.text:
        # remove punctuation and lowercase
        text = re.sub(r'[^a-zA-Z]', ' ', text) 
        text = text.lower()
        
        # tokenize and lemmatize tokens
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(x) for x in tokens]
        text_processed.append(' '.join(tokens))

    # vectorize
    text_matrix = text_vectorizer.transform(text_processed).toarray()
    
    # return np matrix
    return text_matrix

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    d = {'text': [text]}
    # create dataframe from user input
    X_df = pd.DataFrame(data=d)

    # preprocess df and return np array
    X_np = preprocess(X_df)

    # convert to tensor
    X_tensor = torch.Tensor(X_np)

    # predict
    y_pred = model(X_tensor)
    y_pred_max = torch.max(y_pred,1)[1]
    if y_pred_max == 1:
        result = "real"
    else:
        result = "fake"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run()