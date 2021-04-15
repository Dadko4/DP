import numpy as np
import pickle

with open("X.pickle","rb") as f:
    _ = pickle.load(f)
    X_train = pickle.load(f)
    X_val = pickle.load(f)
    X_test = pickle.load(f)
    
from bayes_opt import BayesianOptimization
import lsanomaly
import copy

best_score = 0
best_params = None

def get_diff_score(sigma, rho, n_kernels_max, threshold):
    global best_score, best_params
    n_kernels_max = int(n_kernels_max)
    clf = lsanomaly.LSAnomaly(sigma=sigma, rho=rho, n_kernels_max=n_kernels_max)
    clf.fit(X_train)
    y_val = clf.predict_proba(X_val)[:, 1]
    y_test = clf.predict_proba(X_test)[:, 1]

    len_test = y_test.shape[0]
    len_val = y_val.shape[0]
    TP = y_test[y_test > threshold].shape[0]
    FP = y_val[y_val > threshold].shape[0]
    score = (TP/len_test) - (FP/len_val)
    if score > best_score:
        best_score = score
        best_model = copy.deepcopy(clf)
        best_params = (sigma, rho, n_kernels_max, threshold)
    return score

pbounds = {'sigma': (0.02, 15), 'rho': (0.01, 10), 'n_kernels_max': (100, 50000),
           'threshold': (0.01, 1)}
optimizer = BayesianOptimization(f=get_diff_score, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=5, n_iter=100)
print(best_params)

with open("best_model.pickle", "wb") as f:
    pickle.dump(best_model, f)
