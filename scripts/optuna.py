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


data = pd.read_parquet("../raw data/combined_data_with_embeddings.parquet")
data = pd.concat([data, pd.DataFrame(np.array(data['embedding'].to_list()))], axis = 1)
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'Human' else 1)
data['Label + Dataset'] = data.apply(lambda x:str(x['Label']) + "_" + x['Original dataset'], axis = 1)



def objective(trial):

    classifier_name = trial.suggest_categorical('classifier',['RandomForest', 'AdaBoostClassifier', 'HistGradientBoostingClassifier', 'XGBoost'])
    X = data.loc[:,0:1023]
    y = data['Label']
    
    cv_skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    
    if classifier_name=="RandomForest":
        param = {"n_jobs":30}
        param['max_depth']   = trial.suggest_int('rf_max_depth', 2, 32, log=True)
        param['n_estimators']= trial.suggest_int('rf_n_estimators', 10, 1000, log=True)
        param["max_features"]= trial.suggest_categorical('rf_max_features',['auto', 'sqrt', 'log2'])
        
        classifier_obj = AdaBoostClassifier(**param)
        

    if classifier_name=="AdaBoostClassifier":
        param = {"n_jobs":30}
        param['max_depth']   = trial.suggest_int('ada_max_depth', 2, 32, log=True)
        param['n_estimators']= trial.suggest_int('ada_n_estimators', 10, 1000, log=True)
        param["max_features"]= trial.suggest_categorical('rf_max_features',['auto', 'sqrt', 'log2'])
        
        classifier_obj = RandomForestClassifier(**param)
        
    if classifier_name=="HistGradientBoostingClassifier":
        param = {"n_jobs":30}
        param['max_depth']   = trial.suggest_int('hist_max_depth', 2, 32, log=True)
        param["max_features"]= trial.suggest_categorical('hist_max_features',['auto', 'sqrt', 'log2'])
        
        classifier_obj = HistGradientBoostingClassifier(**param)
    
    if classifier_name=="XGBoost":
        param={"n_jobs":30,
               'eval_metric':'auc',
               'use_label_encoder':False}
        
        param['n_estimators'] = trial.suggest_int('xgb_n_estimators', 10, 1000, log=True)
        param['max_depth']    = trial.suggest_int('xgb_max_depth', 2, 32, log=True)
        param['learning_rate']= trial.suggest_float("xgb_eta", 1e-8, 1.0, log=True)
        param["gamma"]        = trial.suggest_float("xgb_gamma", 1e-8, 1.0, log=True)  
        param['reg_alpha']    = trial.suggest_float("xgb_alpha", 1e-8, 1.0, log=True)
        param['reg_lambda']   = trial.suggest_float("xgb_lambda", 1e-8, 1.0, log=True)
        param['booster']      = trial.suggest_categorical('xgb_booster', ['gbtree', 'gblinear', 'dart'])
    
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
        classifier_obj = xgb.XGBClassifier(**param)
        
        
    cv_results = cross_validate(estimator=classifier_obj,
                                X=X,
                                y=y,
                                scoring='acc',
                                cv=cv_skf)
    print(param)
    print(cv_results)
    mean_score = cv_results['test_score'].mean()
    return mean_score

study = optuna.create_study(study_name="best_clr",
                            direction='maximize',
                            load_if_exists=True,storage="sqlite:///best_clr.db")

study.optimize(objective, n_trials=10)
