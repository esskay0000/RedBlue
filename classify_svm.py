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
from sklearn.feature_extraction import text
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


##FEATURE ENGINEERING##
#Creating an augmented stop words list comprised of 'english' words and candidate and moderator names.
candmodnames = ["donald","trump","ted","cruz","john","kasich","marco","rubio","ben","carson",
                "jeb","bush","chris","christie","carly","fiorina","mike","huckabee","rick","perry",
                "scott","walker","rand","paul","bobby","jindal","lindsey","graham","jim","gilmore",
                "george","pataki","rick","santorum","hillary","clinton","bernie","sanders","martin",
                "o'malley","lincoln","chaffe","jim","webb",'hemmer','maccallum','tapper','bash',
                'hewitt','harwood','quick','quintanilla','regan','seib','smith','blitzer','baier',
                'kelly','wallace','baker','bartiromo','cavuto','muir','raddatz','garrett','strassel',
                'arraras','dinan','cooper','lemon','lopez','cordes','cooney','dickerson','obradovich',
                'holt','mitchell','todd','maddow','ifill','woodruff','ramos','salinas','tumulty','louis']

stop_words = text.ENGLISH_STOP_WORDS
# print stop_words
# print len(stop_words)
# print "\n"
custom_stop_words = text.ENGLISH_STOP_WORDS.union(candmodnames)
# print custom_stop_words
# print len(custom_stop_words)
# print "\n"

# new_words= []
# for i in custom_stop_words:
#     if i not in stop_words:
#         new_words.append(i)
# print new_words
# print len(new_words)

#TfIdfVectorizer = CountVectorizer and TfIdfTransformer all in one step.
tfidf = TfidfVectorizer(stop_words=custom_stop_words)
X_train_tfidf_1 = tfidf.fit_transform(dataset.data)
print "\n Here are the dimensions of our dataset: \n", X_train_tfidf_1.shape, "\n"

#Singular Value Decomposition: Model fit and transform
#This identifies the optimal number of features to select out of all created in the tf-idf vectorizer
#and truncates the vectorizer to make it leaner.

#Without any truncating of the Logistic Regression model (11,288 features), the f1-score = 0.80 and the accuracy_score = 0.82.
#Without any truncating of the Multinomial Naive Bayes model (11,288 features), the f1-score = 0.73 and the accuracy_score = 0.78.
#Without any truncating of the Support Vector Machine model (11,288 features), the f1-score = 0.83 and the accuracy_score = 0.84.

# tsvd      = TruncatedSVD(n_components=100)
# X_train_tfidf_100 = tsvd.fit_transform(X_train_tfidf_1)
# print X_train_tfidf_100.shape
#LR: f1-score = 0.74 and the accuracy_score = 0.77.
#MNB: f1-score = xx and the accuracy_score = xx.
#SVM: f1-score = 0.76 and the accuracy_score = 0.79.

# tsvd      = TruncatedSVD(n_components=1000)
# X_train_tfidf_1000 = tsvd.fit_transform(X_train_tfidf_1)
# print X_train_tfidf_1000.shape
#LR: f1-score = 0.79 and the accuracy_score = 0.81.
#MNB: f1-score = xx and the accuracy_score = xx.
#SVM: f1-score = 0.82 and the accuracy_score = 0.83.

# tsvd      = TruncatedSVD(n_components=1500)
# X_train_tfidf_1500 = tsvd.fit_transform(X_train_tfidf_1)
# print X_train_tfidf_1500.shape
#LR: f1-score = 0.80 and the accuracy_score = 0.82.
#MNB: f1-score = xx and the accuracy_score = xx.
#SVM: f1-score = 0.83 and the accuracy_score = 0.84.

tsvd      = TruncatedSVD(n_components=2000)
X_train_tfidf_2000 = tsvd.fit_transform(X_train_tfidf_1)
print X_train_tfidf_2000.shape
#LR: f1-score = 0.81 and the accuracy_score = 0.83.
#MNB: f1-score = xx and the accuracy_score = xx.
#SVM: f1-score = 0.84 and the accuracy_score = 0.84.

# tsvd      = TruncatedSVD(n_components=3000)
# X_train_tfidf_3000 = tsvd.fit_transform(X_train_tfidf_1)
# print X_train_tfidf_3000.shape
#LR: f1-score = 0.81 and the accuracy_score = 0.83.
#MNB: f1-score = xx and the accuracy_score = xx.
#SVM: f1-score = 0.83 and the accuracy_score = 0.84.

# tsvd      = TruncatedSVD(n_components=4000)
# X_train_tfidf_4000 = tsvd.fit_transform(X_train_tfidf_1)
# print X_train_tfidf_4000.shape
#LR: f1-score = 0.81 and the accuracy_score = 0.83.
#MNB: f1-score = xx and the accuracy_score = xx.
#SVM: f1-score = 0.83 and the accuracy_score = 0.84.

# tsvd      = TruncatedSVD(n_components=5000)
# X_train_tfidf_5000 = tsvd.fit_transform(X_train_tfidf_1)
# print X_train_tfidf_5000.shape
#LR: f1-score = 0.80 and the accuracy_score = 0.82.
#MNB: f1-score = xx and the accuracy_score = xx.
#SVM: f1-score = 0.83 and the accuracy_score = 0.84.

# tsvd      = TruncatedSVD(n_components=8000)
# X_train_tfidf_8000 = tsvd.fit_transform(X_train_tfidf_1)
# print X_train_tfidf_8000.shape
#LR: f1-score = 0.80 and the accuracy_score = 0.82.
#MNB: f1-score = xx and the accuracy_score = xx.
#SVM: f1-score = 0.83 and the accuracy_score = 0.84.

# tsvd      = TruncatedSVD(n_components=11000)
# X_train_tfidf_11000= tsvd.fit_transform(X_train_tfidf_1)
# print X_train_tfidf_11000.shape
#LR: f1-score = 0.81 and the accuracy_score = 0.83.
#MNB: f1-score = xx and the accuracy_score = xx.
#SVM: f1-score = 0.83 and the accuracy_score = 0.84.

#FINAL MODEL GOING FORWARD BASED ON THESE RESULTS (fewest features with the greatest performance):
X_train_tfidf = X_train_tfidf_2000
print X_train_tfidf.shape

##ANALYTICAL MODELING##

#Logistic Regression: Model fit, transform, and testing
splits     = cv.train_test_split(X_train_tfidf, dataset.target, test_size=0.2)
X_train, X_test, Y_train, Y_test = splits

model_lr      = LogisticRegression()
model_lr.fit(X_train, Y_train)

## Variable "expected" is our actual category, dem or rep.
expected   = Y_test

#Variable "predicted" is our model's prediction based on the training data, dem or rep
predicted  = model_lr.predict(X_test)

print "\n Here's our classification report, showing the Logistic Regression model's accuracy: \n"
print classification_report(expected, predicted)
print "Here's a matrix showing results. Clockwise from top left, it's"
print " # correct dem classifications, # incorrect dem, # incorrect rep, # correct rep"
print metrics.confusion_matrix(expected, predicted)
print "\n"
print "Here's the accuracy score for the model: \n"
print metrics.accuracy_score(expected, predicted)

#Multinomial Naive Bayes: Model fit, transform, and testing
#Note that this model could not operate with the SVD, so it relies on the originally
#fitted tf-idf model with 11,228 features.
splits     = cv.train_test_split(X_train_tfidf_1, dataset.target, test_size=0.2)
X_train, X_test, Y_train, Y_test = splits

model_mnb      = MultinomialNB()
model_mnb.fit(X_train, Y_train)

expected   = Y_test
predicted  = model_mnb.predict(X_test)

print "\n Here's our classification report, showing the Multinomial Naive Bayes model's accuracy: \n"
print classification_report(expected, predicted)
print "Here's a matrix showing results. Clockwise from top left, it's"
print " # correct dem classifications, # incorrect dem, # incorrect rep, # correct rep"
print metrics.confusion_matrix(expected, predicted)
print "\n"
print "Here's the accuracy score for the model: \n"
print metrics.accuracy_score(expected, predicted)

#Support Vector Machine: Model fit, transform, and testing

splits     = cv.train_test_split(X_train_tfidf, dataset.target, test_size=0.2)
X_train, X_test, Y_train, Y_test = splits

model_svm      = svm.LinearSVC()
model_svm.fit(X_train, Y_train)

expected   = Y_test
predicted  = model_svm.predict(X_test)

print "\n Here's our classification report, showing the Support Vector Machine model's accuracy: \n"
print classification_report(expected, predicted)
print "Here's a matrix showing results. Clockwise from top left, it's"
print " # correct dem classifications, # incorrect dem, # incorrect rep, # correct rep"
print metrics.confusion_matrix(expected, predicted)
print "\n"
print "Here's the accuracy score for the model: \n"
print metrics.accuracy_score(expected, predicted)



#Saving Fitted Estimators--Support Vector Machine
modelpath = '/Users/Goodgame/desktop/RedBlue/models/' #This should be customized.

with open (modelpath + 'model2000.pkl', 'wb') as f:
    pickle.dump(model_svm, f)

# #Saving the next two because they took a long time (20-40 minutes) to fit and transform. Don't do this unless you want to tie up your CPU for a while.
# with open (path + 'model8000.pkl', 'wb') as f:
#     pickle.dump(X_train_tfidf_8000,f)

# with open (path + 'model11000.pkl', 'wb') as f:
#     pickle.dump(X_train_tfidf_11000,f)

#Loading Fitted Estimator of Interest
with open (path + 'model2000.pkl', 'rb') as f:
    model_svm = pickle.load(f)

# MAKING PREDICTIONS USING RSS FEED OPERATIONAL DATA
# Inserting new data into the SVM model and classifying it

pred_arry = []
array = []

data_dir = '/Users/Goodgame/desktop/RedBlue/output_13apr/'
for doc in os.listdir(output_dir):
    if doc == ".DS_Store": continue # Skip the .DS_Store file
    array.append(doc)
for i in array:
    with codecs.open(output_dir + i, 'r', 'utf-8') as f:
        var = f.read()
        if os.path.getsize(output_dir + i) > 0:
            X_new_tfidf = tfidf.transform(var)
            X_new_tsvd = tsvd.transform(X_new_tfidf)
            predicted = model_svm.predict(X_new_tsvd)
            pred_arry.append(predicted)
            print predicted
print "The number of input documents is ", len(array)
print "The number of test results is ", len(pred_arry)
print array[10]
print pred_arry[10]

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, category))
