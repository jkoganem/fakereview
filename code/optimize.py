print("Importing packages")
import pandas as pd
import numpy as np
import logging
import sys
import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from tqdm import tqdm
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
xgb.set_config(verbosity=0)
print("Done")

print("Reading data")
data = pd.read_parquet("../raw data/embeddings-SBERT.parquet")
data = pd.concat([data, pd.DataFrame(np.array(data['embedding_full'].to_list()))], axis = 1)
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Human' else 1)
data['Label + Dataset'] = data.apply(lambda x:str(x['Label']) + "_" + x['Original dataset'], axis = 1)
print("Done")

le = LabelEncoder()

def objective(trial):
    classifier_name = trial.suggest_categorical(
        'classifier', 
        ['RandomForest', 'AdaBoostClassifier', 'HistGradientBoostingClassifier', 'XGBoost']
    )
    
    X = data.loc[:, 0:767]
    y = le.fit_transform(data['Label'])
    
    cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    if classifier_name == "RandomForest":
        param = {
            "n_jobs": 1,
            "max_depth": trial.suggest_int('rf_max_depth', 2, 32, log=True),
            "n_estimators": trial.suggest_int('rf_n_estimators', 10, 1000, log=True),
            "max_features": trial.suggest_categorical('rf_max_features', ['auto', 'sqrt', 'log2'])
        }
        classifier_obj = RandomForestClassifier(**param)
    
    elif classifier_name == "AdaBoostClassifier":
        param = {
            "n_estimators": trial.suggest_int('ada_n_estimators', 10, 1000, log=True),
            "learning_rate": trial.suggest_float("ada_learning_rate", 0.01, 1.0, log=True)
        }
        classifier_obj = AdaBoostClassifier(**param)
    
    elif classifier_name == "HistGradientBoostingClassifier":
        param = {
            "max_iter": trial.suggest_int('hist_max_iter', 10, 100),
            "max_depth": trial.suggest_int('hist_max_depth', 2, 32, log=True),
            "learning_rate": trial.suggest_float("hist_learning_rate", 0.01, 1.0, log=True)
        }
        classifier_obj = HistGradientBoostingClassifier(**param)
    
    elif classifier_name == "XGBoost":
        param = {
            "n_jobs": 1,
            "eval_metric": 'auc',
            "use_label_encoder": False,
            "n_estimators": trial.suggest_int('xgb_n_estimators', 10, 1000, log=True),
            "max_depth": trial.suggest_int('xgb_max_depth', 2, 32, log=True),
            "learning_rate": trial.suggest_float("xgb_eta", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("xgb_gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("xgb_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("xgb_lambda", 1e-8, 1.0, log=True),
            "booster": trial.suggest_categorical('xgb_booster', ['gbtree', 'gblinear', 'dart'])
        }
        
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
        classifier_obj = xgb.XGBClassifier(**param)
    
    cv_results = cross_validate(
        estimator=classifier_obj,
        X=X,
        y=y,
        scoring='accuracy',
        cv=cv_skf
    )
    
    print(param)
    print(cv_results)
    mean_score = cv_results['test_score'].mean()
    return mean_score

study = optuna.create_study(study_name="best_clr",
                            direction='maximize',
                            load_if_exists=True,storage="sqlite:///best_clr.db")
print("Starting trials")
study.optimize(objective, n_trials=10, show_progress_bar = True)
print("Done")