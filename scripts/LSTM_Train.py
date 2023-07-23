import numpy as np
import pandas as pd
import nltk
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tokenized():
  '''
  Uses Tensorflow Tokenizer to filter out punctuation and split into words with space, apply texts_to_sequences to get matric representation
  '''
  raw_df = pd.read_csv('data/raw_df.csv')
  tokenizer = Tokenizer(num_words = 3000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower = True, split = ' ')
  tokenizer.fit_on_texts(texts = raw_df['full_text'])
  X = tokenizer.texts_to_sequences(texts = raw_df['full_text'])
  X = pad_sequences(sequences = X, maxlen = 3000, padding = 'pre')
  y = raw_df['label'].values
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)
  return X_train, X_test, y_train, y_test

def lstm_model():
  '''
  Structure a lstm model
  '''
  lstm_model = Sequential(name = 'lstm_nn_model')
  lstm_model.add(layer = Embedding(input_dim = 3000, output_dim = 120, name = '1st_layer'))
  lstm_model.add(layer = LSTM(units = 120, dropout = 0.2, recurrent_dropout = 0.2, name = '2nd_layer'))
  lstm_model.add(layer = Dropout(rate = 0.5, name = '3rd_layer'))
  lstm_model.add(layer = Dense(units = 120,  activation = 'relu', name = '4th_layer'))
  lstm_model.add(layer = Dropout(rate = 0.5, name = '5th_layer'))
  lstm_model.add(layer = Dense(units = len(set(y)),  activation = 'sigmoid', name = 'output_layer'))
  # compiling the model
  lstm_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
  return lstm_model

if __name__ == "__main__":
  X_train, X_test, y_train, y_test = get_tokenized()
  model = lstm_model()
  lstm_model_fit = model.fit(X_train, y_train, epochs = 1)
  y_pred = (model.predict(X_test) >= 0.5).astype("int")
  y_pred = y_pred[:,1]
  score = accuracy_score(y_test,y_pred)
  print(f'Accuracy on test set: {round(score*100,2)}%')
     
     
