#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pickle
from bayes_opt import BayesianOptimization
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
# import lsanomaly
import copy
np.random.seed(1)

# In[2]:


with open("X.pickle","rb") as f:
    scaler = pickle.load(f)
    X_train = pickle.load(f)
    X_val = pickle.load(f)
    X_test = pickle.load(f)
    
idx = np.random.randint(0, X_test.shape[0], X_test.shape[0])

X_val = X_val[idx]
X_test = X_test[idx]

best_score = 0
best_params = None
best_model = None
y_val = None
y_test = None


# In[3]:


def get_diff_score(kernel, degree, tol, nu, shrinking):
    global best_score, best_params, best_model, y_val, y_test
    
    kernel_map_vals = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel_map = {i: v for i, v in enumerate(kernel_map_vals)}
    
    kernel = kernel_map[int(kernel)]
    degree = int(degree)
    shrinking = bool(round(shrinking))
    
    clf = OneClassSVM(kernel=kernel, degree=degree, shrinking=shrinking,
                      cache_size=70000)
    clf.fit(X_train)
    
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(clf.score_samples(X_train).reshape(-1, 1)).reshape(-1)
    y_val = scaler.transform(clf.score_samples(X_val).reshape(-1, 1)).reshape(-1)
    y_test = scaler.transform(clf.score_samples(X_test).reshape(-1, 1)).reshape(-1)
    
    threshold = np.percentile(y_train, 50)
    
    len_val = y_val.shape[0]
    len_test = y_test.shape[0]
    
    TP = y_test[y_test < threshold].shape[0]
    FP = y_val[y_val < threshold].shape[0]
    TN = len_val - FP
    
    acc = (TP + TN) / (len_val + len_test)
    precision = TP / (TP + FP)
    recall = TP / len_test
    f1 = (2*precision*recall) / (precision + recall)
    
    score = 100 * acc
    
    print(f"precision {precision}, recall {recall}")
    print(f"acc {acc}")
    
    if score > best_score:
        best_score = score
        best_model = copy.deepcopy(clf)
        best_params = best_model.get_params()
    
    return score


# In[4]:


pbounds = {'kernel': (-0.5, 3.49), 'degree': (2, 10), 'tol': (1e-5, 1e-7),
           'nu': (0.001, 1), 'shrinking': (0, 1)}
optimizer = BayesianOptimization(f=get_diff_score, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=15, n_iter=50)
print(best_params)

with open("best_model_svm.pickle", "wb") as f:
    pickle.dump(best_model, f)

