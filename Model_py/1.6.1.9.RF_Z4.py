#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
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
        'n_estimators': [300, 500, 1000], #4, 8, 16, 32, 64,
        'criterion' : ['gini', 'entropy'],
        'max_features': ["sqrt", "log2"],
        'random_state': [0]
    }

    rf_model = RandomForestClassifier()
    grid_search = GridSearchCV(rf_model, param_grid=fr_param_grid, cv=5, return_train_score = True, scoring='roc_auc', n_jobs=24)
    grid_search.fit(train_x, train_y)
    best_params = grid_search.best_params_
    best_model = RandomForestClassifier(**best_params)

    best_model.fit(train_x, train_y)
    y_pred_4 = best_model.predict(test_x)
    filename = output_name1 + '.sav'
    pickle.dump(best_model, open(filename, 'wb'))

    print('AUC: %.2f' % roc_auc_score(test_y, best_model.predict_proba(test_x)[:,1]))

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
        'n_estimators': [300, 500, 1000], #4, 8, 16, 32, 64,
        'criterion' : ['gini', 'entropy'],
        'max_features': ["sqrt", "log2"],
        'random_state': [0]
    }

    rf_model = RandomForestClassifier()
#    grid_search = GridSearchCV(rf_model, param_grid=fr_param_grid, cv=custom_cv, return_train_score = True, scoring='roc_auc', n_jobs=1)
    grid_search = GridSearchCV(rf_model, param_grid=fr_param_grid, cv=5, return_train_score = True, scoring='roc_auc', n_jobs=24)
    grid_search.fit(train_x, train_y)
    best_params = grid_search.best_params_
    best_model = RandomForestClassifier(**best_params)

    best_model.fit(train_x, train_y)
    y_pred_4 = best_model.predict(test_x)
    filename = output_name1 + '.sav'
    pickle.dump(best_model, open(filename, 'wb'))
    print('AUC: %.2f' % roc_auc_score(test_y, best_model.predict_proba(test_x)[:,1]))

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

#outer_path = [zeevi_path]
outer_path = [serum_path, baxter_path, zeevi_path]
outer_path = [zeevi_path]
fname_lists = ['no_psm', '1_psm', '2_psm', '3_psm', '4_psm']


# In[ ]:


result = pd.DataFrame()
result2 = pd.DataFrame()
for d_path in outer_path:
    for seed_num in range(41,51):
        seed_path = d_path + 'seed' + str(seed_num) + '/'
        test_fname = seed_path + 'Ts_clr_py.csv'
        test = pd.read_csv(test_fname, index_col=0)
        print(seed_path)
        for file_idx in range(0,5):
            Tr_addr = seed_path + 'Tr_' + fname_lists[file_idx] + '_no_smoted'
            train = pd.read_csv(Tr_addr + '.csv', index_col=0)
            Feature_addr = seed_path + 'UNIV_' + fname_lists[file_idx] + '_top_list_L6.txt'
            MathcingTable = pd.read_csv(Feature_addr)
            feature_lists = cov + MathcingTable['x'].values.flatten().tolist()
            output1 = RF_run(train, test, feature_lists, Tr_addr)
            result = result.append([output1])

            output2 = RF_run2(train, test, Tr_addr)
            result2 = result2.append([output2])
            for SCALE in SCALE_lists:
                euc_path = seed_path + 'Tr_' + fname_lists[file_idx] + '_euc_smoted_' + 'scale_' + str(SCALE)
                mah_path = seed_path + 'Tr_' + fname_lists[file_idx] + '_mah_smoted_' + 'scale_' + str(SCALE)
                bc_path = seed_path + 'Tr_' + fname_lists[file_idx] + '_bc_smoted_' + 'scale_' + str(SCALE)
                uu_path = seed_path + 'Tr_' + fname_lists[file_idx] + '_uu_smoted_' + 'scale_' + str(SCALE)
                gu_path = seed_path + 'Tr_' + fname_lists[file_idx] + '_gu_smoted_' + 'scale_' + str(SCALE)
                wu_path = seed_path + 'Tr_' + fname_lists[file_idx] + '_wu_smoted_' + 'scale_' + str(SCALE)
                euc_train = pd.read_csv(euc_path + '.csv', index_col=0)
                mah_train = pd.read_csv(mah_path + '.csv', index_col=0)
                bc_train = pd.read_csv(bc_path + '.csv', index_col=0)
                uu_train = pd.read_csv(uu_path + '.csv', index_col=0)
                gu_train = pd.read_csv(gu_path + '.csv', index_col=0)
                wu_train = pd.read_csv(wu_path + '.csv', index_col=0)
                output1 = RF_run(euc_train, test, feature_lists, euc_path)
                result = result.append([output1])
                output1 = RF_run(mah_train, test, feature_lists, mah_path)
                result = result.append([output1])
                output1 = RF_run(bc_train, test, feature_lists, bc_path)
                result = result.append([output1])
                output1 = RF_run(uu_train, test, feature_lists, uu_path)
                result = result.append([output1])
                output1 = RF_run(gu_train, test, feature_lists, gu_path)
                result = result.append([output1])
                output1 = RF_run(wu_train, test, feature_lists, wu_path)
                result = result.append([output1])
                output2 = RF_run2(euc_train, test, euc_path)
                result2 = result2.append([output2])
                output2 = RF_run2(mah_train, test, mah_path)
                result2 = result2.append([output2])
                output2 = RF_run2(bc_train, test, bc_path)
                result2 = result2.append([output2])
                output2 = RF_run2(uu_train, test, uu_path)
                result2 = result2.append([output2])
                output2 = RF_run2(gu_train, test, gu_path)
                result2 = result2.append([output2])
                output2 = RF_run2(wu_train, test, wu_path)
                result2 = result2.append([output2])


result.to_csv(path + 'Thesis/univ_markers_RF_9_Z_4.csv')
result2.to_csv(path + 'Thesis/total_maseed_pathrkers_RF_9_Z_4.csv')

for d_path in outer_path:
    for seed_num in range(41,51):
        seed_path = d_path + 'seed' + str(seed_num) + '/'
        commands = "mkdir " + seed_path + "RF & mv " + seed_path + "*.sav " + seed_path + "RF/"
        os.system(commands)
