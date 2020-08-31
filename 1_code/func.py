# Linden Parkes, 2020
# lindenmp@seas.upenn.edu

# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import nibabel as nib

# Stats
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import pingouin as pg

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

# Extra
from scipy.linalg import svd, schur
from numpy.matlib import repmat
from statsmodels.stats import multitest

# Sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR



def set_proj_env(parc_str = 'schaefer', parc_scale = 200, edge_weight = 'streamlineCount', extra_str = ''):

    # Project root directory
    projdir = '/Users/lindenmp/Google-Drive-Penn/work/research_projects/neurodev_cs_predictive'; os.environ['PROJDIR'] = projdir

    # Data directory
    datadir = os.path.join(projdir, '0_data'); os.environ['DATADIR'] = datadir

    # Imaging derivatives
    derivsdir = os.path.join('/Volumes/work_ssd/research_data/PNC/'); os.environ['DERIVSDIR'] = derivsdir

    # Pipeline directory
    pipelinedir = os.path.join(projdir, '2_pipeline'); os.environ['PIPELINEDIR'] = pipelinedir

    # Output directory
    outputdir = os.path.join(projdir, '3_output'); os.environ['OUTPUTDIR'] = outputdir

    # Parcellation specifications
    # Names of parcels
    if parc_str == 'schaefer': parcel_names = np.genfromtxt(os.path.join(projdir, 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames.txt'), dtype='str')
    # vector describing whether rois belong to cortex (1) or subcortex (0)
    if parc_str == 'schaefer': parcel_loc = np.loadtxt(os.path.join(projdir, 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames_loc.txt'), dtype='int')

    drop_parcels = []
    num_parcels = parcel_names.shape[0]

    if parc_str == 'schaefer':
        scdir = os.path.join(derivsdir, 'processedData/diffusion/deterministic_20171118'); os.environ['SCDIR'] = scdir
        sc_name_tmp = 'bblid/*xscanid/tractography/connectivity/bblid_*xscanid_SchaeferPNC_' + str(parc_scale) + '_dti_' + edge_weight + '_connectivity.mat'; os.environ['SC_NAME_TMP'] = sc_name_tmp

        os.environ['CONN_STR'] = 'connectivity'

        rstsdir = os.path.join(derivsdir, 'processedData/restbold/restbold_201607151621'); os.environ['RSTSDIR'] = rstsdir
        if parc_scale == 200:
            rsts_name_tmp = 'bblid/*xscanid/net/Schaefer' + str(parc_scale) + 'PNC/bblid_*xscanid_Schaefer' + str(parc_scale) + 'PNC_ts.1D'; os.environ['RSTS_NAME_TMP'] = rsts_name_tmp
        elif parc_scale == 400:
            rsts_name_tmp = 'bblid/*xscanid/net/SchaeferPNC/bblid_*xscanid_SchaeferPNC_ts.1D'; os.environ['RSTS_NAME_TMP'] = rsts_name_tmp

    return parcel_names, parcel_loc, drop_parcels, num_parcels


def my_get_cmap(which_type = 'qual1', num_classes = 8):
    # Returns a nice set of colors to make a nice colormap using the color schemes
    # from http://colorbrewer2.org/
    #
    # The online tool, colorbrewer2, is copyright Cynthia Brewer, Mark Harrower and
    # The Pennsylvania State University.

    if which_type == 'linden':
        cmap_base = np.array([[255,105,97],[97,168,255],[178,223,138],[117,112,179],[255,179,71]])
    elif which_type == 'pair':
        cmap_base = np.array([[124,230,199],[255,169,132]])
    elif which_type == 'qual1':
        cmap_base = np.array([[166,206,227],[31,120,180],[178,223,138],[51,160,44],[251,154,153],[227,26,28],
                            [253,191,111],[255,127,0],[202,178,214],[106,61,154],[255,255,153],[177,89,40]])
    elif which_type == 'qual2':
        cmap_base = np.array([[141,211,199],[255,255,179],[190,186,218],[251,128,114],[128,177,211],[253,180,98],
                            [179,222,105],[252,205,229],[217,217,217],[188,128,189],[204,235,197],[255,237,111]])
    elif which_type == 'seq_red':
        cmap_base = np.array([[255,245,240],[254,224,210],[252,187,161],[252,146,114],[251,106,74],
                            [239,59,44],[203,24,29],[165,15,21],[103,0,13]])
    elif which_type == 'seq_blu':
        cmap_base = np.array([[247,251,255],[222,235,247],[198,219,239],[158,202,225],[107,174,214],
                            [66,146,198],[33,113,181],[8,81,156],[8,48,107]])
    elif which_type == 'redblu_pair':
        cmap_base = np.array([[222,45,38],[49,130,189]])
    elif which_type == 'yeo17':
        cmap_base = np.array([[97,38,107], # VisCent
                            [194,33,39], # VisPeri
                            [79,130,165], # SomMotA
                            [44,181,140], # SomMotB
                            [75,148,72], # DorsAttnA
                            [23,116,62], # DorsAttnB
                            [149,77,158], # SalVentAttnA
                            [222,130,177], # SalVentAttnB
                            [75,87,61], # LimbicA
                            [149,166,110], # LimbicB
                            [210,135,47], # ContA
                            [132,48,73], # ContB
                            [92,107,131], # ContC
                            [218,221,50], # DefaultA
                            [175,49,69], # DefaultB
                            [41,38,99], # DefaultC
                            [53,75,158] # TempPar
                            ])
    elif which_type == 'yeo17_downsampled':
        cmap_base = np.array([[97,38,107], # VisCent
                            [79,130,165], # SomMotA
                            [75,148,72], # DorsAttnA
                            [149,77,158], # SalVentAttnA
                            [75,87,61], # LimbicA
                            [210,135,47], # ContA
                            [218,221,50], # DefaultA
                            [53,75,158] # TempPar
                            ])

    if cmap_base.shape[0] > num_classes: cmap = cmap_base[0:num_classes]
    else: cmap = cmap_base

    cmap = cmap / 255

    return cmap


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return sp.stats.norm.ppf(x)


def rank_int(series, c=3.0/8):
    # Check input
    assert(isinstance(series, pd.Series))
    assert(isinstance(c, float))

    # Set seed
    np.random.seed(123)

    # Drop NaNs
    series = series.loc[~pd.isnull(series)]

    # Get rank, ties are averaged
    rank = sp.stats.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
    
    return transformed


def node_strength(A):
    s = np.sum(A, axis = 0)

    return s


def ave_control(A, c = 1):
    # FUNCTION:
    #         Returns values of AVERAGE CONTROLLABILITY for each node in a
    #         network, given the adjacency matrix for that network. Average
    #         controllability measures the ease by which input at that node can
    #         steer the system into many easily-reachable states.
    #
    # INPUT:
    #         A is the structural (NOT FUNCTIONAL) network adjacency matrix, 
    #         such that the simple linear model of dynamics outlined in the 
    #         reference is an accurate estimate of brain state fluctuations. 
    #         Assumes all values in the matrix are positive, and that the 
    #         matrix is symmetric.
    #
    # OUTPUT:
    #         Vector of average controllability values for each node
    #
    # Bassett Lab, University of Pennsylvania, 2016.
    # Reference: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia,
    #            Vettel, Miller, Grafton & Bassett, Nature Communications
    #            6:8414, 2015.

    u, s, vt = svd(A) # singluar value decomposition
    A = A/(c + s[0]) # Matrix normalization 
    T, U = schur(A,'real') # Schur stability
    midMat = np.multiply(U,U).transpose()
    v = np.matrix(np.diag(T)).transpose()
    N = A.shape[0]
    P = np.diag(1 - np.matmul(v,v.transpose()))
    P = repmat(P.reshape([N,1]), 1, N)
    values = sum(np.divide(midMat,P))
    
    return values


def get_fdr_p(p_vals, alpha = 0.05):
    out = multitest.multipletests(p_vals, alpha = alpha, method = 'fdr_bh')
    p_fdr = out[1] 

    return p_fdr


def get_fdr_p_df(p_vals, alpha = 0.05, rows = False):
    
    if rows:
        p_fdr = pd.DataFrame(index = p_vals.index, columns = p_vals.columns)
        for row, data in p_vals.iterrows():
            p_fdr.loc[row,:] = get_fdr_p(data.values)
    else:
        p_fdr = pd.DataFrame(index = p_vals.index,
                            columns = p_vals.columns,
                            data = np.reshape(get_fdr_p(p_vals.values.flatten(), alpha = alpha), p_vals.shape))

    return p_fdr


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


def get_reg(num_params = 10):
    regs = {'rr': Ridge(),
            'lr': Lasso(),
            'krr_lin': KernelRidge(kernel='linear'),
            'krr_rbf': KernelRidge(kernel='rbf'),
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


def get_stratified_cv(X, y, c = None, n_splits = 10):

    # sort data on outcome variable in ascending order
    idx = y.sort_values(ascending = True).index
    if X.ndim == 2: X_sort = X.loc[idx,:]
    elif X.ndim == 1: X_sort = X.loc[idx]
    y_sort = y.loc[idx]
    if c is not None:
        if c.ndim == 2: c_sort = c.loc[idx,:]
        elif c.ndim == 1: c_sort = c.loc[idx]
    
    # create custom stratified kfold on outcome variable
    my_cv = []
    for k in range(n_splits):
        my_bool = np.zeros(y.shape[0]).astype(bool)
        my_bool[np.arange(k,y.shape[0],n_splits)] = True

        train_idx = np.where(my_bool == False)[0]
        test_idx = np.where(my_bool == True)[0]
        my_cv.append( (train_idx, test_idx) )

    if c is not None:
        return X_sort, y_sort, my_cv, c_sort
    else:
        return X_sort, y_sort, my_cv


def cross_val_score_nuis(X, y, c, my_cv, reg, my_scorer):
    
    accuracy = np.zeros(len(my_cv),)
    y_pred_out = np.zeros(y.shape)
    
    for k in np.arange(len(my_cv)):
        tr = my_cv[k][0]
        te = my_cv[k][1]

        # Split into train test
        X_train = X.iloc[tr,:]; X_test = X.iloc[te,:]
        y_train = y.iloc[tr]; y_test = y.iloc[te]
        c_train = c.iloc[tr,:]; c_test = c.iloc[te,:]

        # standardize predictors
        sc = StandardScaler(); sc.fit(X_train); X_train = sc.transform(X_train); X_test = sc.transform(X_test)
        X_train = pd.DataFrame(data = X_train, index = X.iloc[tr,:].index, columns = X.iloc[tr,:].columns)
        X_test = pd.DataFrame(data = X_test, index = X.iloc[te,:].index, columns = X.iloc[te,:].columns)

        # standardize covariates
        sc = StandardScaler(); sc.fit(c_train); c_train = sc.transform(c_train); c_test = sc.transform(c_test)
        c_train = pd.DataFrame(data = c_train, index = c.iloc[tr,:].index, columns = c.iloc[tr,:].columns)
        c_test = pd.DataFrame(data = c_test, index = c.iloc[te,:].index, columns = c.iloc[te,:].columns)

        # regress nuisance (X)
        nuis_reg = LinearRegression(); nuis_reg.fit(c_train, X_train)
        X_pred = nuis_reg.predict(c_train); X_train = X_train - X_pred
        X_pred = nuis_reg.predict(c_test); X_test = X_test - X_pred

        # # regress nuisance (y)
        # nuis_reg = LinearRegression(); nuis_reg.fit(c_train, y_train)
        # y_pred = nuis_reg.predict(c_train); y_train = y_train - y_pred
        # y_pred = nuis_reg.predict(c_test); y_test = y_test - y_pred

        reg.fit(X_train, y_train)
        accuracy[k] = my_scorer(reg, X_test, y_test)
        
        y_pred_out[te] = reg.predict(X_test)
        
    return accuracy, y_pred_out


def assemble_df(numpy_array, algs, metrics, phenos):
    df = pd.DataFrame(columns = ['score', 'alg', 'metric', 'pheno'])

    for a, alg in enumerate(algs):
        for m, metric in enumerate(metrics):
            for p, pheno in enumerate(phenos):
                df_tmp = pd.DataFrame(columns = df.columns)
                df_tmp.loc[:,'score'] = numpy_array[:,a,m,p]
                df_tmp.loc[:,'alg'] = alg
                df_tmp.loc[:,'metric'] = metric
                df_tmp.loc[:,'pheno'] = pheno

                df = pd.concat((df, df_tmp), axis = 0)
    
    return df


def get_exact_p(x,y):
    pval = 2*np.min([np.mean(x-y>=0), np.mean(x-y<=0)])
    
    return pval
