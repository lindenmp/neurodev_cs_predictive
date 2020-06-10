import argparse

# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import copy
import json

# Stats
import scipy as sp
from scipy import stats

# Sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-x", help="IVs", dest="X_file", default=None)
parser.add_argument("-y", help="DVs", dest="y_file", default=None)
parser.add_argument("-c", help="DVs", dest="c_file", default=None)
parser.add_argument("-metric", help="brain feature (e.g., ac)", dest="metric", default=None)
parser.add_argument("-pheno", help="psychopathology dimension", dest="pheno", default=None)
parser.add_argument("-seed", help="seed for shuffle_data", dest="seed", default=1)
parser.add_argument("-alg", help="estimator", dest="alg", default=None)
parser.add_argument("-score", help="score set order", dest="score", default=None)
parser.add_argument("-o", help="output directory", dest="outroot", default=None)

args = parser.parse_args()
print(args)
X_file = args.X_file
y_file = args.y_file
c_file = args.c_file
metric = args.metric
pheno = args.pheno
# seed = int(args.seed)
# seed = int(os.environ['SGE_TASK_ID'])-1
alg = args.alg
score = args.score
outroot = args.outroot
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# prediction functions
def corr_true_pred(y_true, y_pred):
    if type(y_true) == np.ndarray:
        y_true = y_true.flatten()
    if type(y_pred) == np.ndarray:
        y_pred = y_pred.flatten()
        
    r,p = sp.stats.pearsonr(y_true, y_pred)
    return r


def root_mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2, axis=0)
    rmse = np.sqrt(mse)
    return rmse


def shuffle_data(X, y, c, seed = 0):
    np.random.seed(seed)
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)

    X_shuf = X.iloc[idx,:]
    y_shuf = y.iloc[idx]
    c_shuf = c.iloc[idx,:]
    
    return X_shuf, y_shuf, c_shuf


def get_reg(num_params = 10):
    regs = {'rr': Ridge(max_iter = 100000),
            'lr': Lasso(max_iter = 100000),
            'krr_lin': KernelRidge(kernel='linear'),
            'krr_rbf': KernelRidge(kernel='rbf'),
            # 'svr_lin': LinearSVR(max_iter=100000),
            'svr_lin': SVR(kernel='linear'),
            'svr_rbf': SVR(kernel='rbf')
            }
    
    # From the sklearn docs, gamma defaults to 1/n_features. In my cases that will be either 1/400 features = 0.0025 or 1/200 = 0.005.
    # I'll set gamma to same range as alpha then [0.001 to 1] - this way, the defaults will be included in the gridsearch
    param_grids = {'rr': {'reg__alpha': np.logspace(0, -3, num_params)},
                    'lr': {'reg__alpha': np.logspace(0, -3, num_params)},
                   'krr_lin': {'reg__alpha': np.logspace(0, -3, num_params)},
                   'krr_rbf': {'reg__alpha': np.logspace(0, -3, num_params), 'reg__gamma': np.logspace(0, -3, num_params)},
                    'svr_lin': {'reg__C': np.logspace(0, 4, num_params)},
                    'svr_rbf': {'reg__C': np.logspace(0, 4, num_params), 'reg__gamma': np.logspace(0, -3, num_params)}
                    }
    
    return regs, param_grids


def get_cv(y, n_splits = 10):

    my_cv = []

    kf = KFold(n_splits = 10, shuffle = False)

    for train_idx, test_idx in kf.split(y):
        my_cv.append( (train_idx, test_idx) )

    return my_cv


def cross_val_score_nuis(X, y, c, my_cv, reg, my_scorer, c_y = None):
    
    accuracy = np.zeros(len(my_cv),)

    for k in np.arange(len(my_cv)):
        tr = my_cv[k][0]
        te = my_cv[k][1]

        # Split into train test
        X_train = X.iloc[tr,:]; X_test = X.iloc[te,:]
        y_train = y.iloc[tr].values.reshape(-1,1); y_test = y.iloc[te].values.reshape(-1,1)
        c_train = c.iloc[tr,:]; c_test = c.iloc[te,:]
        if c_y is not None: c_y_train = c_y.iloc[tr,:]; c_y_test = c_y.iloc[te,:]

        # standardize predictors
        sc = StandardScaler(); sc.fit(X_train); X_train = sc.transform(X_train); X_test = sc.transform(X_test)

        # standardize covariates
        sc = StandardScaler(); sc.fit(c_train); c_train = sc.transform(c_train); c_test = sc.transform(c_test)
        if c_y is not None: sc = StandardScaler(); sc.fit(c_y_train); c_y_train = sc.transform(c_y_train); c_y_test = sc.transform(c_y_test)

        # regress nuisance (X)
        nuis_reg = LinearRegression(); nuis_reg.fit(c_train, X_train)
        X_pred = nuis_reg.predict(c_train); X_train = X_train - X_pred
        X_pred = nuis_reg.predict(c_test); X_test = X_test - X_pred

        # regress nuisance (y)
        if c_y is None:  
            nuis_reg = LinearRegression(); nuis_reg.fit(c_train, y_train)
            y_pred = nuis_reg.predict(c_train); y_train = y_train - y_pred
            y_pred = nuis_reg.predict(c_test); y_test = y_test - y_pred
        elif c_y is not None:
            nuis_reg = LinearRegression(); nuis_reg.fit(c_y_train, y_train)
            y_pred = nuis_reg.predict(c_y_train); y_train = y_train - y_pred
            y_pred = nuis_reg.predict(c_y_test); y_test = y_test - y_pred

        reg.fit(X_train, y_train)
        accuracy[k] = my_scorer(reg, X_test, y_test)
        
    return accuracy



def run_reg(X, y, c, reg, my_scorer, n_splits = 10, seed = 0):

    X_shuf, y_shuf, c_shuf = shuffle_data(X = X, y = y, c = c, seed = seed)

    my_cv = get_cv(y_shuf, n_splits = n_splits)

    accuracy = cross_val_score_nuis(X = X_shuf, y = y_shuf, c = c_shuf, my_cv = my_cv, reg = reg, my_scorer = my_scorer)

    return accuracy


# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# inputs
X = pd.read_csv(X_file)
X.set_index(['bblid', 'scanid'], inplace = True)
X = X.filter(regex = metric)

y = pd.read_csv(y_file)
y.set_index(['bblid', 'scanid'], inplace = True)
y = y.loc[:,pheno]

c = pd.read_csv(c_file)
c.set_index(['bblid', 'scanid'], inplace = True)

# outdir
outdir = os.path.join(outroot, score, alg + '_' + metric + '_' + pheno)
if not os.path.exists(outdir): os.makedirs(outdir);
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# set scorer
if score == 'r2':
    my_scorer = make_scorer(r2_score, greater_is_better = True)
elif score == 'corr':
    my_scorer = make_scorer(corr_true_pred, greater_is_better = True)
elif score == 'mse':
    my_scorer = make_scorer(mean_squared_error, greater_is_better = False)
elif score == 'rmse':
    my_scorer = make_scorer(root_mean_squared_error, greater_is_better = False)
elif score == 'mae':
    my_scorer = make_scorer(mean_absolute_error, greater_is_better = False)

# prediction
regs, param_grids = get_reg()

num_random_splits = 50

accuracy_mean = np.zeros(num_random_splits)
accuracy_std = np.zeros(num_random_splits)

for i in np.arange(0,num_random_splits):
    accuracy = run_reg(X = X, y = y, c = c, reg = regs[alg], my_scorer = my_scorer, seed = i)
    accuracy_mean[i] = accuracy.mean()
    accuracy_std[i] = accuracy.std()
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# outputs
np.savetxt(os.path.join(outdir,'accuracy_mean.txt'), accuracy_mean)
np.savetxt(os.path.join(outdir,'accuracy_std.txt'), accuracy_std)

# --------------------------------------------------------------------------------------------------------------------

print('Finished!')
