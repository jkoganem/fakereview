### Importing Modules ###

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import torch
import nltk
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import os.path
from tqdm.autonotebook import tqdm, trange
from sklearn.svm import SVC
from datetime import datetime

### Loading data with embeddings ###
print('Loading data with embeddings...')
df = pd.read_parquet("../raw data/combined_data_with_embeddings.parquet")

### Make reviews and wiki the test and train data respectively ###
train = df[df['Original dataset'] == 'reviews']
test = df[df['Original dataset'] == 'wiki']

X_train = np.vstack(train.embedding.apply(lambda x: np.asarray(x).flatten()))
X_test = np.vstack(test.embedding.apply(lambda x: np.asarray(x).flatten()))

### Training model ###

from sklearn.svm import SVC
clf = SVC(kernel='linear')

print('Training...')
clf.fit(X_train, train.Label)

print('Training complete.')

### Check accuracy score ###
preds = clf.predict(X_test)

ac = accuracy_score(test.Label, preds)
cm = confusion_matrix(test.Label, preds)

print('The model\'s accuracy score is ', ac)
print ('The confusion matrix is as follows:', cm)
