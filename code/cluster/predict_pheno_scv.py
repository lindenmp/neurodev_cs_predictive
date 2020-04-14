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
from sklearn.linear_model import Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error

# --------------------------------------------------------------------------------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-x", help="IVs", dest="X_file", default=None)
parser.add_argument("-y", help="DVs", dest="y_file", default=None)
parser.add_argument("-metric", help="brain feature (e.g., ac)", dest="metric", default=None)
parser.add_argument("-pheno", help="psychopathology dimension", dest="pheno", default=None)
parser.add_argument("-seed", help="seed for shuffle_data", dest="seed", default=None)
parser.add_argument("-alg", help="estimator", dest="alg", default=None)
parser.add_argument("-score", help="score set order", dest="score", default=None)
parser.add_argument("-o", help="output directory", dest="outroot", default=None)

args = parser.parse_args()
print(args)
X_file = args.X_file
y_file = args.y_file
metric = args.metric
pheno = args.pheno
# seed = int(args.seed)
seed = int(os.environ['SGE_TASK_ID'])-1
alg = args.alg
score = args.score
outroot = args.outroot
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# prediction functions
def corr_pred_true(y_pred, y_true):
    r = sp.stats.pearsonr(y_pred, y_true)[0]
    return r

my_scorer = make_scorer(corr_pred_true, greater_is_better = True)


def get_reg(num_params = 10):
    regs = {'rr': Ridge(),
            'lr': Lasso(),
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


def get_stratified_cv(X, y, n_splits = 10):

    # sort data on outcome variable in ascending order
    idx = y.sort_values(ascending = True).index
    X_sort = X.loc[idx,:]
    y_sort = y.loc[idx]
    
    # create custom stratified kfold on outcome variable
    my_cv = []
    for k in range(n_splits):
        my_bool = np.zeros(y.shape[0]).astype(bool)
        my_bool[np.arange(k,y.shape[0],n_splits)] = True

        train_idx = np.where(my_bool == False)[0]
        test_idx = np.where(my_bool == True)[0]
        my_cv.append( (train_idx, test_idx) )  

    return X_sort, y_sort, my_cv


def run_reg_scv(X, y, reg, param_grid, n_splits = 10, scoring = 'r2'):
    
    pipe = Pipeline(steps=[('standardize', StandardScaler()),
                           ('reg', reg)])
    
    X_sort, y_sort, my_cv = get_stratified_cv(X, y, n_splits = n_splits)
    
    # if scoring is a dictionary then we run GridSearchCV with multiple scoring metrics and refit using the first one in the dict
    if type(scoring) == dict: grid = GridSearchCV(pipe, param_grid, cv = my_cv, scoring = scoring, refit = list(scoring.keys())[0])
    else: grid = GridSearchCV(pipe, param_grid, cv = my_cv, scoring = scoring)
    
    grid.fit(X_sort, y_sort);
    
    return grid


# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# inputs
X = pd.read_csv(X_file)
X.set_index(['bblid', 'scanid'], inplace = True)
X = X.filter(regex = metric)

y = pd.read_csv(y_file)
y.set_index(['bblid', 'scanid'], inplace = True)
y = y.loc[:,pheno]

# outdir
outdir = os.path.join(outroot, 'main_score_' + score, alg + '_' + metric + '_' + pheno)
if not os.path.exists(outdir): os.makedirs(outdir);
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# set scorer
if score == 'r2':
    # use R2 as main metric
    scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'corr': my_scorer}
elif score == 'corr':
    # use corr as main metric
    scoring = {'corr': my_scorer, 'r2': 'r2', 'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
elif score == 'mse':
    # use mse as main metric
    scoring = {'mse': 'neg_mean_squared_error', 'r2': 'r2', 'mae': 'neg_mean_absolute_error', 'corr': my_scorer}

# prediction
regs, param_grids = get_reg()

grid = run_reg_scv(X = X, y = y, reg = regs[alg], param_grid = param_grids[alg], scoring = scoring)

best_params = grid.best_params_
if type(scoring) == dict:
    best_scores = dict()
    for key in scoring.keys():
        best_scores[key] = grid.cv_results_['mean_test_'+key][grid.best_index_]
else:
    best_scores = grid.best_score_
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# outputs
json_data = json.dumps(best_params)
f = open(os.path.join(outdir,'best_params.json'),'w')
f.write(json_data)
f.close()

json_data = json.dumps(best_scores)
f = open(os.path.join(outdir,'best_scores.json'),'w')
f.write(json_data)
f.close()

# --------------------------------------------------------------------------------------------------------------------

print('Finished!')
