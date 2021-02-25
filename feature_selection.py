#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:48:46 2021

function: feature selection for genotype prediction in glioma

@author: zst
"""



import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    
    # load data set
    datapath = 'xxx/features_tumor_all_t1.csv'  # an example: prediction of 1p/19q using t1 features
    datapath_edema = 'xxx/features_edema_all_t1.csv'
    infopath = 'xxx/info_tumor.csv'
    featuredata = pd.read_csv(datapath, header=0)
    featuredata = featuredata.iloc[:,14:]
    featuredata_edema = pd.read_csv(datapath_edema,header=0)
    featuredata_edema = featuredata_edema.iloc[:,14:]
    featuredata_edema.columns = featuredata_edema.columns+'_edema'
    featuredata = pd.concat([featuredata, featuredata_edema], axis=1)
    featuredata = featuredata.fillna(0)
    infodata = pd.read_csv(infopath, header=0)
    
    # split dataset
    idx_random = 97
    X_train, X_test = train_test_split(featuredata, test_size=0.3333, random_state=idx_random)
    feaName = list(X_train)
    assert feaName == list(X_test), print('the features in the training and test set are not the same')
    info_train, info_test = train_test_split(infodata, test_size=0.3333, random_state=idx_random)
    y_train, y_test = info_train['1p19q.codeletion'], info_test['1p19q.codeletion']

    fea_name = list(X_train)

    ###############################################################################
    ######################### feature selection####################################
    ###############################################################################
    from scipy.stats import kruskal
    # 1 select features which are robust among different MR machine
    value_p = []
    for i in range(pd.np.shape(X_train)[1]):
        group0 = pd.np.where(info_train['M']==0)
        group0 = X_train.iloc[group0[0], i]
        group1 = pd.np.where(info_train['M']==1)
        group1 = X_train.iloc[group1[0], i]
        group2 = pd.np.where(info_train['M']==2)
        group2 = X_train.iloc[group2[0], i]
        _, p = kruskal(group0,group1, group2)
        value_p.append(p)
    value_p = np.asarray(value_p)
    X_train = X_train.iloc[:, np.where(value_p>0.05)[0]]
    X_test = X_test.iloc[:, np.where(value_p>0.05)[0]]
    value_p = value_p[np.where(value_p>0.05)[0]]
    
    
    from scipy.stats import mannwhitneyu as utest
    # 2 select features which are robust between 1.5 and 3 T MR images
    value_p = []
    for i in range(pd.np.shape(X_train)[1]):
        group0 = pd.np.where(info_train['B']==0)
        group0 = X_train.iloc[group0[0], i]
        group1 = pd.np.where(info_train['B']==1)
        group1 = X_train.iloc[group1[0], i]
        _, p = kruskal(group0,group1)
        value_p.append(p)
    value_p = np.asarray(value_p)
    X_train = X_train.iloc[:, np.where(value_p>0.05)[0]]
    X_test = X_test.iloc[:, np.where(value_p>0.05)[0]]
    value_p = value_p[np.where(value_p>0.05)[0]]
    
    from sklearn.feature_selection import VarianceThreshold    
    # 3 remove features with low variance
    fs_sel = VarianceThreshold(threshold=1)
    fs_sel.fit_transform(X_train)
    X_train = X_train.iloc[:,fs_sel.get_support()]
    X_test = X_test.iloc[:,fs_sel.get_support()]
    fea_name = list(X_train)
    
    # feature normalization with z-score normalization method
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(data=X_train, columns=fea_name)
    X_test = pd.DataFrame(data=X_test, columns=fea_name)
    print(len(fea_name))
    
    # 4 select features which are significatly associated with label
    value_p = []
    for i in range(pd.np.shape(X_train)[1]):
        group1 = np.where(y_train==0)
        group1 = X_train.iloc[group1[0],i]
        group2 = np.where(y_train==1)
        group2 = X_train.iloc[group2[0],i]
        _, p = utest(group1, group2)
        value_p.append(p)
    value_p = np.asarray(value_p)
    data_train1 = X_train.iloc[:, np.where(value_p<0.05)[0]]
    data_test1 = X_test.iloc[:, np.where(value_p<0.05)[0]]
    value_p = value_p[np.where(value_p<0.05)[0]]

    # 5 remove features with high redundance using correlation coefficient
    
    from sklearn.feature_selection import RedundancyThresholdSurv
    value_ptmp = 1-value_p
    corrcoef = np.abs(np.corrcoef(data_train1, rowvar=0))
    corrcoef = np.triu(corrcoef)
    while 1:
        selector_redundancy = RedundancyThresholdSurv(threshold=0.8)
        selector_redundancy.fit(data_train1, value_ptmp)
        data_redundancy_train = data_train1.iloc[:, selector_redundancy.get_support()]
        data_redundancy_test = data_test1.iloc[:,selector_redundancy.get_support()]
        value_ptmp = value_ptmp[selector_redundancy.get_support()]
        if data_train1.equals(data_redundancy_train):
            break
        data_train1 = data_redundancy_train
        data_test1 = data_redundancy_test

    list_featurename = list(data_train1)
    X_train = data_train1
    X_test = data_test1
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Lasso
    # 6 select the best feature set using lasso
    cv = 10
    cls_ = 'lasso'
    parameters = {'alpha':np.logspace(-1.3, -1.0, 200)}
    model = Lasso(max_iter=1000)
    clf = GridSearchCV(model, parameters, cv=cv, refit=True)
    clf.fit(X_train, y_train)
    X_train1 = X_train.iloc[:,clf.best_estimator_.coef_!=0]
    X_test1 = X_test.iloc[:,clf.best_estimator_.coef_!=0]
    # write select feature into csv files
#    X_train1.to_csv('xxx/pq_t1_train1.csv', index=False)
#    X_test1.to_csv('xxx/pq_t1_test1.csv', index=False)        
    