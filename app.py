from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pandas as pd
import keras
import pickle
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding,SpatialDropout1D
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import re
import nltk
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))


def clean_text(text):
    print(text)
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    print(text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


app = Flask(__name__)
model = keras.models.load_model("./hate&abusive_model.h5")
with open('tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        hatespeech = request.form["hate"]
        hatespeech = [clean_text(hatespeech)]
        seq = load_tokenizer.texts_to_sequences(hatespeech)
        padded = sequence.pad_sequences(seq, maxlen=300)
        pred = model.predict(padded)


        if pred<0.2:
            
            return render_template('home.html',prediction_text="NO hate {}".format(pred))
        
        else:

            return render_template('home.html',prediction_text=" Hate and Abusive {}".format(pred))


    return render_template("home.html")



if __name__ == "__main__":
    app.run(debug=True)


        




