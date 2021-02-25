#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:42:26 2020

@author: Shuaitong Zhang
"""

import numpy as np
from ..base import BaseEstimator
from .base import SelectorMixin
from ..utils import check_array
from ..utils.validation import check_is_fitted

"""note: place this file in the directory of /xxx/python3.6/site-packages/sklearn/feature_selection/ and
   modify the __init__.py file"""

class RedundancyThresholdSurv(BaseEstimator, SelectorMixin):
    """Feature selector that removes features with high redundancy.

    This feature selection algorithm groups features with high relevance at the features (X), then select
    an optimal one (feature with highest Cindex) from each group.
    
    notice: This function needs the C-index of each single feature

    Parameters
    ----------
    threshold : float, optional
        Features with correlation higher than this threshold will
        be grouped. The default is 0.75.
    
    Example
    ----------
    from sklearn.feature_selection import RedundancyThresholdSurv
    
    redundance_sel = RedundancyThresholdSurv(threshold=0.75)
    redundance_sel.fit(X_train, y=auc_train) #auc_train: list, auc value for each single feature
    X_train = X_train.iloc[:,redundance_sel.get_support()]  
    X_test = X_test.iloc[:,redundance_sel.get_support()]

    """

    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Sample vectors from which to compute correlation coefficient.

        y : C-indcies for all features.

        Returns
        -------
        self
        """
        X = check_array(X, ('csr', 'csc'), dtype=np.float64)
        y = np.asarray(y)
		
        if hasattr(X, "toarray"):   # sparse matrix
            msg = "the format of X is sparse matrix, which is not support by this function. Please convert X to matrix"
            raise ValueError(msg)
		#_, self.variances_ = mean_variance_axis(X, axis=0)
        else:
            self.corrcoef_ = np.abs(np.corrcoef(X, rowvar=0))

        if np.all(self.corrcoef_ <= self.threshold):
            msg = "Correlation coeficients are lower than threshold {0:.5f} for all features in X"
            print(msg.format(self.threshold))
            self.featureselectindex_ = np.array(map(bool,range(1,np.shape(X)[1]+1)))
            return self
        if np.all(self.corrcoef_ >= self.threshold):
            msg = "Correlation coeficients are higher than threshold {0:.5f} for all features in X"
            if X.shape[0] == 1:
                msg += " (X contains only one sample)"
                raise ValueError(msg.format(self.threshold))
            msg += " (only one feature was selected)"
            self.featureselectindex_ = np.array(map(bool,np.zeros((np.shape(X)[1]))))
            self.featureselectindex_[self.mostrelevantfeature(self, range(np.shape(X)[1]))] = True
            print(msg.format(self.threshold))
			
            return self
			
		
        self.featuregroup_ = []
#        self.corrcoef_ = np.triu(self.corrcoef_)
        for i in self.corrcoef_:
            self.featuregroup_.append(np.where(i>=self.threshold)[0])
        self.featureselect_ = []
        for i in self.featuregroup_:
            self.featureselect_.append(self.mostrelevantfeature(i,X,y))
#        self.featureselect_ = map(self.mostrelevantfeature(self,X=X,y=y),self.featuregroup_)
        self.featureselect_ = list(set(self.featureselect_))
        self.featureselectindex_ = np.array(list(map(bool,np.zeros((np.shape(X)[1])))))
        self.featureselectindex_[self.featureselect_] = True
        return self
        
    def mostrelevantfeature(self, a, X, y): # a: a list, which contains some feature index in self.featuregroup_
        cindextmp = []
        for i in a:
            cindextmp.append(y[i])
        return a[np.argmax(np.abs(np.asarray(cindextmp)))]

	
    def _get_support_mask(self):
        check_is_fitted(self, 'featureselectindex_')

        return self.featureselectindex_
