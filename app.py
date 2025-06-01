import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

##getting indexes for one hot representation
word_index=imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

##loading model
model=load_model('simpe_rnn.h5')

##helper fnc

##decode review
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

##adding padding and preprocessing
def preprocess_text(text):
  words=text.lower().split()
  encoded_review=[word_index.get(word,2) + 3 for word in words]
  padded_review= sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review


##prediction fnc
def predict_senti(review):
  preprocessed_inp=preprocess_text(review)

  prediction=model.predict(preprocessed_inp)

  sentiment='Pos' if prediction[0][0] > 0.5 else 'Neg'

  return sentiment , prediction[0][0]

##setting p the app
import streamlit as st

st.title('IMDB REVIEW CLASSIFICATION')
st.write('Enter a movie review to classify it as positive and negative')

user_input=st.text_input('Enter your review')

if st.button('Classify'):
  preprocess_inp=preprocess_text(user_input)
  prediction=model.predict(preprocess_inp)
  sentiment='Pos' if prediction[0][0] > 0.5 else 'Neg'
  score=prediction[0][0]
  st.write(f'the score is {score}')
  st.write(f'The review is {sentiment}')

else:
  st.write('enter your review')
