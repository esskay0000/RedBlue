import os
import time
import pickle

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
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.svm import SVC
from sklearn import svm
from sklearn import grid_search
from sklearn.feature_extraction import text
import lxml
import pprint

## For data exploration
import pandas as pd
import numpy as np
import pandas.io.sql as pd_sql
import sqlite3 as sql

# For encoding
import codecs

# For visualization
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

 ###############################################################################
 # This script takes the picked models from classify_svm.py and applies them
 # to directories of text data (here, using RSS feeds transformed to text)
 ###############################################################################


#Loading Fitted Estimator of Interest
modelpath = '/Users/Goodgame/desktop/RedBlue/models/' #This should be customized.

with open (modelpath + 'tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open (modelpath + 'tsvd.pkl', 'rb') as f:
    tsvd = pickle.load(f)

with open (modelpath + 'model_svm.pkl', 'rb') as f:
    model_svm = pickle.load(f)



# Take text-formatted RSS feeds, open them for reading, append them to
# an array, and make predictions on the array

def get_instances_from_files(path):
    docs = []

    for doc in os.listdir(path):

        if doc.startswith('.'): continue # Ignore hidden files
        # Open the file for reading
        with codecs.open(os.path.join(path, doc), 'r', 'utf-8') as f:
            data = f.read() # f is the file handle, now data is the string
            docs.append(data)

    # Now that we're done going through all the files, give back docs
    return docs

dataset = get_instances_from_files('/Users/Goodgame/desktop/RedBlue/data/sources/text_17may/wsj_text/')
X = tfidf.transform(dataset)
X_new_tsvd = tsvd.transform(X)
preds = model_svm.predict(X_new_tsvd)

print "Here are the results, printed in an array:"
pprint.pprint(preds)
print "Here are the number of predictions in the array:"
print len(preds)


#For each news source, where there is an output of values in "array:"

reparray= []
for i,j in enumerate(preds):
	if j == 'rep':
		reparray.append(i)

rep = len(reparray)
rep = float(rep)
print "Here is the number of red results: ", rep


demarray= []
for i,j in enumerate(preds):
	if j == 'dem':
		demarray.append(i)

dem = len(demarray)
dem = float(dem)
print "Here is the number of blue results: ", dem

#Calcualte the distribution a news source's documents' predicted values:

percentrep = (rep/len(preds))*100
percentdem = (dem/len(preds))*100

print "Percentage of red results: %", percentrep
print "Percentage of blue results: %", percentdem
