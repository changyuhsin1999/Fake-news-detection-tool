import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def get_dataset():
    '''
    Loads dataset from .csv file and return as train set and test set.
    '''
    raw_df = pd.read_csv('data/raw_df.csv')
    raw_df = raw_df.dropna()
    raw_df['full_text'] = raw_df.apply(lambda x: ' '.join([x['title'],x['text']]),axis=1)
    labels=raw_df.label
    x_train,x_test,y_train,y_test=train_test_split(raw_df['full_text'], labels, test_size=0.2, random_state=7)
    return x_train, x_test, y_train, y_test

def Tokenized_data():
    '''
    Uses TFIDF as tokenizer to generate tokenized train set nd test set
    '''
    #Initialize a TfidfVectorizer
    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.5)
    #Fit and transform train set, transform test set
    tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test=tfidf_vectorizer.transform(x_test)
    return tfidf_train, tfidf_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_dataset()
    tfidf_train, tfidf_test = Tokenized_data()
    pac=PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train,y_train)
    #Predict on the test set and calculate accuracy
    y_pred=pac.predict(tfidf_test)
    score=accuracy_score(y_test,y_pred)
    print(f'Accuracy on test set: {round(score*100,2)}%')
