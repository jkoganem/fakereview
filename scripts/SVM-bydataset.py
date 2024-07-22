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

for dataset in df['Original dataset'].unique():
    newdf = df[df['Original dataset'] == dataset]

    train, test = train_test_split(newdf, test_size=0.2)

    X_train = np.vstack(train.embedding.apply(lambda x: np.asarray(x).flatten()))
    X_test = np.vstack(test.embedding.apply(lambda x: np.asarray(x).flatten()))
    
    ### Training model ###
    
    clf = SVC(kernel='linear')

    print(f'Training on {dataset}')
    start = datetime.now()
    clf.fit(X_train, train.Label)
    end = datetime.now()
    print(f'Training complete. The training time was {(end-start).total_seconds()} seconds.')
    
    ### Check accuracy score ###
    preds = clf.predict(X_test)
    
    ac = accuracy_score(test.Label, preds)
    cm = confusion_matrix(test.Label, preds)
    
    print(f'The model\'s accuracy score on {dataset} is {ac}')
    print (f'The confusion matrix for {dataset} is as follows:', cm)
