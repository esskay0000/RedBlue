import os
import time

import bs4
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

import codecs

# For troubleshooting
from IPython import embed

import fileinput

CORPUS_ROOT = "/Users/Goodgame/desktop/RedBlue/test_input/"

# for i in os.open(CORPUS_ROOT, os.O_RDONLY):

arry = []
x = 1

for fn in os.listdir(CORPUS_ROOT):
    document = open(CORPUS_ROOT + fn)
    soup = bs4.BeautifulSoup(document.read(), "html.parser")
    for tag in soup.find_all('p'):
        arry.append(tag)
        fname = "%s.txt" % (x)
        outpath = os.path.abspath('test_output/' + fname)
        for line in arry:
            with open(outpath, 'w') as f:
                f.write(line.text.encode('utf-8') + '\n')
        x += 1
    print fn

inputfile = os.path.abspath('test_output/')
outputfile  = os.path.abspath('test_output/aggregated/')

filenames = []
for fn in os.listdir(inputfile):
    filenames.append(fn)

with open(inputfile + 'aggregated.txt', 'w') as fout, fileinput.input(filenames) as fin:
    for line in fin:
        fout.write(line)



## FOR LOOP TO ITERATE THROUGH RSS DATA.
#
# for fn in os.listdir(CORPUS_ROOT):
#     document = open(CORPUS_ROOT + fn)
#     soup = bs4.BeautifulSoup(document.read(), "html.parser")
#     for tag in soup.find_all('p'):
#         txtarry.append(tag)
# for line in txtarry:
#     fname = "%s.txt" % (x)
#     outpath = os.path.abspath('test_output/' + fname)
#     with open(outpath, 'w') as f:
#         f.write('%s\n' % line.text.encode('utf-8'))
#     x += 1
#     txtarry = []
#     print fn
#
# ## TEMPLATE CODE
#
# for i in DEMdebates2016list:
#     soup = BeautifulSoup(urllib2.urlopen(i), "lxml")
#     for tag in soup.find_all('p'):
#         arry.append(tag)
# for line in arry:
#     fname = "dem_%s.txt" % (x)
#     outpath = os.path.abspath(fname)
#     with open(outpath, 'w') as f:
#         f.write(line.text.encode('utf-8') + '\n')
#     x+=1
#
#
#     # f.write(var.text.encode('utf-8') + '\n')

        # for i in txtarry:
        #     a = str(i)
        #     # var = ''.join(a)
        # fname = "%s.txt" % (x)
        # np.savetxt(fname, txtarry)
        #
        #



#



# Working code for printing p tags
# for fn in os.listdir(CORPUS_ROOT):
#     document = open(CORPUS_ROOT + fn)
#     soup = bs4.BeautifulSoup(document.read(), "html.parser")
#     for tag in soup.find_all('p'):
#         txtarry.append(tag)
#         for i in txtarry:
#             a = str(i)
#             var = ''.join(a)
#         print var
