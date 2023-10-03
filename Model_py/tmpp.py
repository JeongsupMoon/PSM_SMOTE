#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, roc_curve, auc, classification_report, precision_recall_curve, confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import itertools
import pydotplus
from IPython.display import Image
import pickle

def RF_run(train, test, feature_lists, output_name1):

    train_x = train[feature_lists]
    train_y = train['class']
    test_x = test[feature_lists]
    test_y = test['class']

    train_y_01=np.array(train_y)
    test_y_01=np.array(test_y)

    fr_param_grid = {
        'C':[0.1,0.5,0.8,1,2,4,8,16,20,25,30],
        'gamma':[0.01, 0.1],
        'kernel': ['rbf'],
        'degree': [2,3,4],
        'random_state': [0],
        'cache_size': [7000]#,'probability': [False]
}

    rf_model = SVC()
    grid_search = GridSearchCV(rf_model, param_grid=fr_param_grid, cv=5, return_train_score = True, scoring='roc_auc', n_jobs=80)
    grid_search.fit(train_x, train_y)
    best_params = grid_search.best_params_
    best_model = SVC(**best_params, probability=True)

    best_model.fit(train_x, train_y)
    y_pred_4 = best_model.predict(test_x)
    filename = output_name1 + '.sav'
    pickle.dump(best_model, open(filename, 'wb'))

    fpr, tpr, thresholds = roc_curve(test_y_01, best_model.predict_proba(test_x)[:,1])
    roc_auc = auc(fpr, tpr)
    spc = 1 - fpr
    valid_idx = np.where(spc >= 0.8)[0]
    out_idx = valid_idx[spc[valid_idx].argmin()]
    ix = np.nanargmin(np.where(spc > 0.8, spc, 1))
    y_pred_char = np.where(y_pred_4 > thresholds[ix], 1, 0)
    output = pd.DataFrame()
    output = output.append(best_params, ignore_index=True)
    output['name'] = [output_name1]
    output.set_index('name', inplace = True)
    output['train_auc'] = [grid_search.cv_results_['mean_train_score'][grid_search.best_index_]]
    output['cv_auc'] = [grid_search.cv_results_['mean_test_score'][grid_search.best_index_]]
    output['test_auc'] = [roc_auc]
    output['sensitivity'] = [tpr[out_idx]]
    output['specificity'] = [spc[out_idx]]
    output['accuracy'] = accuracy_score(test_y, y_pred_char)
    output['balanced accuracy'] = balanced_accuracy_score(test_y, y_pred_char)
    output['precision'] = precision_score(test_y, y_pred_char, average="binary", pos_label=1)
    output['recall'] = recall_score(test_y, y_pred_char, average="binary", pos_label=1)
    output['f1'] = f1_score(test_y, y_pred_char, average="binary", pos_label=1)
    output['mcc'] = matthews_corrcoef(test_y, y_pred_char)
    return(output)

def RF_run2(train, test, output_name1):

    train_x = train.drop('class', axis=1)
    train_y = train['class']
    test_x = test.drop('class', axis=1)
    test_y = test['class']

    train_y_01=np.array(train_y)
    test_y_01=np.array(test_y)

    fr_param_grid = {
        'C':[0.1,0.5,0.8,1,2,4,8,16,20,25,30],
        'gamma':[0.01, 0.1],
        'kernel': ['rbf'],
        'degree': [2,3,4],
        'random_state': [0],
        'cache_size': [7000]#,'probability': [False]
}

    rf_model = SVC()
    grid_search = GridSearchCV(rf_model, param_grid=fr_param_grid, cv=5, return_train_score = True, scoring='roc_auc', n_jobs=80)
    grid_search.fit(train_x, train_y)
    best_params = grid_search.best_params_
    best_model = SVC(**best_params, probability=True)

    best_model.fit(train_x, train_y)
    y_pred_4 = best_model.predict(test_x)
    filename = output_name1 + '.sav'
    pickle.dump(best_model, open(filename, 'wb'))

    fpr, tpr, thresholds = roc_curve(test_y_01, best_model.predict_proba(test_x)[:,1])
    roc_auc = auc(fpr, tpr)

    spc = 1 - fpr
    valid_idx = np.where(spc >= 0.8)[0]
    out_idx = valid_idx[spc[valid_idx].argmin()]

    ix = np.nanargmin(np.where(spc > 0.8, spc, 1))
    y_pred_char = np.where(y_pred_4 > thresholds[ix], 1, 0)

    output = pd.DataFrame()
    output = output.append(best_params, ignore_index=True)
    output['name'] = [output_name1]
    output.set_index('name', inplace = True)
    output['train_auc'] = [grid_search.cv_results_['mean_train_score'][grid_search.best_index_]]
    output['cv_auc'] = [grid_search.cv_results_['mean_test_score'][grid_search.best_index_]]
    output['test_auc'] = [roc_auc]
    output['sensitivity'] = [tpr[out_idx]]
    output['specificity'] = [spc[out_idx]]
    output['accuracy'] = accuracy_score(test_y, y_pred_char)
    output['balanced accuracy'] = balanced_accuracy_score(test_y, y_pred_char)
    output['precision'] = precision_score(test_y, y_pred_char, average="binary", pos_label=1)
    output['recall'] = recall_score(test_y, y_pred_char, average="binary", pos_label=1)
    output['f1'] = f1_score(test_y, y_pred_char, average="binary", pos_label=1)
    output['mcc'] = matthews_corrcoef(test_y, y_pred_char)

    return(output)


# # Data Loading

# In[12]:


cov = ['Age', 'Sex']
y_class = 'class'
Standardization = True
SELF_DIST = 9999
KNN_number = 5
SEED = 777
SCALE_lists = [1, 1.5, 2, 3]

path = "/home2/jsmoon/data/"

serum_path = path + 'Thesis/Serum/'
baxter_path = path + 'Thesis/Baxter/'
zeevi_path = path + 'Thesis/Zeevi/'

outer_path = [serum_path, baxter_path, zeevi_path]
outer_path = [baxter_path]
fname_lists = ['no_psm', '1_psm', '2_psm', '3_psm', '4_psm']

for d_path in outer_path:
    for seed_num in range(11,12):
        seed_path = d_path + 'seed' + str(seed_num) + '/'
        commands = "mkdir " + seed_path + "SVM & mv " + seed_path + "*.sav " + seed_path + "SVM/"
        os.system(commands)
