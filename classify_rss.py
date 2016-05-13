import os
import time


from bs4 import BeautifulSoup
from sklearn.datasets.base import Bunch
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import cross_validation as cv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import lxml

## For data exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas.io.sql as pd_sql
import sqlite3 as sql

# For encoding
import codecs

# For visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#CONNECTING TO THE DATASET
CORPUS_ROOT = "/Users/Goodgame/desktop/RedBlue/debate_data"
#You will have to insert your own path to the transcript data folder here.#

def load_data(root=CORPUS_ROOT):
    """
    Loads the text data into memory using the bundle dataset structure.
    Note that on larger corpora, memory safe CorpusReaders should be used.
    """

    # Open the README and store
    with open(os.path.join(root, 'README'), 'r') as readme:
        DESCR = readme.read()

    # Iterate through all the categories
    # Read the HTML into the data and store the category in target
    data      = []
    target    = []
    filenames = []

    for category in os.listdir(root):
        if category == "README": continue # Skip the README
        if category == ".DS_Store": continue # Skip the .DS_Store file
        for doc in os.listdir(os.path.join(root, category)):
            fname = os.path.join(root, category, doc)

            # Store information about document
            filenames.append(fname)
            target.append(category)
            with codecs.open(fname, 'r', 'ISO-8859-1') as f:
                data.append(f.read())
            # Read data and store in data list
            # with open(fname, 'r') as f:
            #     data.append(f.read())

    return Bunch(
        data=data,
        target=target,
        filenames=filenames,
        target_names=frozenset(target),
        DESCR=DESCR,
    )

dataset = load_data()

#print out the readme file
print dataset.DESCR
#Remember to create a README file and place it inside your CORPUS ROOT directory if you haven't already done so.

#print the number of records in the dataset
print "The number of instances is ", len(dataset.data), "\n"

#Checking out the data
print "Here are the last five instances: ", dataset.data[-5:], "\n"
print "Here are the categories of the last five instances: \n", dataset.target[-5:],
"\n\n"

#CONSTRUCT FEATURE EXTRACTION
#TfIdfVectorizer = CountVectorizer and TfIdfTransformer all in one step.
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(dataset.data)
print "\n Here are the dimensions of our dataset: \n", X_train_tfidf.shape, "\n"

#Logistic Regression: Model fit, transform, and testing
splits     = cv.train_test_split(X_train_tfidf, dataset.target, test_size=0.2)
X_train, X_test, Y_train, Y_test = splits

model      = LogisticRegression()
model.fit(X_train, Y_train)

## Variable "expected" is our actual category, dem or rep.
expected   = Y_test

#Variable "predicted" is our model's prediction based on the training data, dem or rep
predicted  = model.predict(X_test)

print "\n Here's our classification report, showing the model's accuracy: \n"
print classification_report(expected, predicted)
print "Here's a matrix showing results. Clockwise from top left, it's"
print " # correct dem classifications, # incorrect dem, # incorrect rep, # correct rep"
print metrics.confusion_matrix(expected, predicted)
print "\n"

## Inserting new data into the model and classifying it

pred_arry = []
array = []

output_dir = '/Users/Goodgame/desktop/RedBlue/output_13apr/'
for doc in os.listdir(output_dir):
    if doc == ".DS_Store": continue # Skip the .DS_Store file
    array.append(doc)
for i in array:
    with codecs.open(output_dir + i, 'r', 'utf-8') as f:
        var = f.read()
        if os.path.getsize(output_dir + i) > 0:
            X_new_tfidf = tfidf.transform(var)
            predicted = model.predict(X_new_tfidf)
            pred_arry.append(predicted)
            print predicted
print "The number of input documents is ", len(array)
print "The number of test results is ", len(pred_arry)
print array[10]
print pred_arry[10]

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, category))
