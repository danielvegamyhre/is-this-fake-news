import re
import pickle
import numpy as np
import pandas as pd
import torch
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request



MODEL_PATH = 'data/model.pkl'
with open(MODEL_PATH, 'rb') as fp:
    model = pickle.load(fp)

def preprocess(df):
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
 

    title_processed = []
    for title in df.title:
        # remove punctuation and lowercase
        title = re.sub(r'[^a-zA-Z]', ' ', title) 
        title = title.lower()
        
        # tokenize and lemmatize tokens
        tokens = word_tokenize(title)
        tokens = [lemmatizer.lemmatize(x) for x in tokens]
        title_processed.append(' '.join(tokens))
        
    # vectorize
    text_vectorizer = CountVectorizer(stop_words='english', max_features=4000)
    title_vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    text_matrix = text_vectorizer.fit_transform(text_processed).toarray()
    title_matrix = title_vectorizer.fit_transform(title_processed).toarray()
    
    # store label then drop old text columns and label
    y = np.array(df.label)
    df.drop(['title','text','label'], inplace=True, axis=1)
    
    # return np matrix
    X = np.concatenate([np.array(df), title_matrix, text_matrix], axis=1)
    return X, y

app = Flask(__name__)

@app.route('/')
def home():
    pass

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    text = request.form['text']
    d = {'title': [title], 'text': [text]}
    df = pd.DataFrame(data=d)
    X = preprocess(df)
    y_pred = model(X)
    y_pred_max = torch.max(y_pred,1)[1]
    if y_pred_max == 0:
        my_prediction = "Real news!"
    else:
        my_prediction = "Fake news!"
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run()