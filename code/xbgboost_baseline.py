## For data handling
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm.notebook import tqdm

## Metrics
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix

## For plotting
import matplotlib.pyplot as plt
from seaborn import set_style
set_style("whitegrid")

## Parameters 
EMBEDDING_LENGTH = 383

# Import our data
data = pd.read_parquet("../raw data/embeddings-SBERT.parquet")

# Put embedding coords into separate columns for ease of fitting below
data = pd.concat([data, pd.DataFrame(np.array(data['embedding_light'].to_list()))], axis = 1)

# Set integer labels
# Human = 0, 1 = Machine
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Human' else 1)

# Train / test split stratified by label and original dataset
data['Label + Dataset'] = data.apply(lambda x:str(x['Label']) + "_" + x['Original dataset'], axis = 1) 
train, test = train_test_split(data, 
                               stratify = data['Label + Dataset'],
                               random_state = 406,
                               train_size = 0.8)

# 5-fold cross validation, stratified by label and original dataset
kfold = StratifiedKFold(n_splits = 5,
                        shuffle = True,
                        random_state = 406)

acc_scores = []

for fold, (t_index, v_index) in enumerate(kfold.split(train, train['Label + Dataset'])):
    # Get training x and labels, and validation x and labels
    tt_x = (train.iloc[t_index]).loc[:, 0:383]
    tt_y = (train.iloc[t_index])['Label']
    vv_x = (train.iloc[v_index]).loc[:, 0:383]
    vv_y = (train.iloc[v_index])['Label']
    print(f"Fitting fold {fold}")
    xgb = xgboost.XGBClassifier()
    xgb.fit(tt_x.values, tt_y.values.reshape(-1,1))
    print(f"Predicting fold {fold}")
    pred = xgb.predict(vv_x)
    acc_scores += [acc(vv_y.values.reshape(-1,1), pred)]

print("Final accuracy scores for each fold:")
print(acc_scores)

# xgboost on full training data with many estimators
tt, vv = train_test_split(train,
                          stratify = train['Label + Dataset'],
                          train_size = 0.8,
                          random_state = 406)
tt_x = tt.loc[:, 0:EMBEDDING_LENGTH]
tt_y = tt['Label']
vv_x = vv.loc[:, 0:EMBEDDING_LENGTH]
vv_y = vv['Label']
xgb = xgboost.XGBClassifier(n_estimators = 1500,
                           max_depth = 5,
                           random_state = 406)
xgb.fit(tt_x.values, tt_y.values.reshape(-1,1))
pred = xgb.predict(vv_x)
print(f"Accuracy: {acc(vv_y.values.reshape(-1,1), pred)}")
print(f"Confusion matrix: \n{confusion_matrix(vv_y.values.reshape(-1,1), pred, normalize = 'true')}")