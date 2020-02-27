#!/usr/bin/env python
# coding: utf-8

# # Results, section 2:

# In[1]:


import os, sys
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error


# In[2]:


sys.path.append('/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec_DWI/code/func/')
from proj_environment import set_proj_env
from func import mark_outliers, get_cmap, run_corr, get_fdr_p, perc_dev, evd, summarise_network


# In[3]:


exclude_str = 't1Exclude'
parc_str = 'schaefer'
parc_scale = 200
primary_covariate = 'ageAtScan1'
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(exclude_str = exclude_str,
                                                                            parc_str = parc_str, parc_scale = parc_scale,
                                                                            primary_covariate = primary_covariate)


# In[4]:


os.environ['NORMATIVEDIR']


# In[5]:


metrics = ('ct', 'vol', 'str', 'ac', 'mc')
phenos = ('Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear',
         'F1_Exec_Comp_Res_Accuracy', 'F2_Social_Cog_Accuracy', 'F3_Memory_Accuracy', 'F1_Complex_Reasoning_Efficiency',
          'F2_Memory.Efficiency', 'F3_Executive_Efficiency', 'F4_Social_Cognition_Efficiency',)


# ## Setup plots

# In[6]:


if not os.path.exists(os.environ['FIGDIR']): os.makedirs(os.environ['FIGDIR'])
os.chdir(os.environ['FIGDIR'])
sns.set(style='white', context = 'talk', font_scale = 1)
cmap = sns.color_palette("pastel", 3)


# ## Load data

# In[7]:


df = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'df.csv'))
df.set_index(['bblid', 'scanid'], inplace = True); print(df.shape)

df_node = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'df_node.csv'))
df_node.set_index(['bblid', 'scanid'], inplace = True); print(df_node.shape)

df_z = pd.read_csv(os.path.join(os.environ['NORMATIVEDIR'], 'df_z.csv'))
df_z.set_index(['bblid', 'scanid'], inplace = True); print(df_node.shape)


# # Predictive model

# In[17]:


def run_krr(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0, shuffle = True)
    print(X_test.shape)

    pipe = Pipeline(steps=[('standardize', StandardScaler()),
                           ('pca', PCA()),
                           ('reg', KernelRidge(kernel='rbf'))])

    param_grid = {'pca__n_components': [10, 15, 20],
                  'reg__alpha': np.logspace(-5, 5, 5),
                  'reg__gamma': np.logspace(-5, 5, 5)}

    # scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
    # scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'}

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
    grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring = 'r2')
    grid.fit(X_train, y_train);

    print(grid.best_score_)
    print(grid.best_params_)

#     outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)
#     nested_score = cross_val_score(grid, X=X_train, y=y_train, cv=outer_cv)
#     print('Nested CV score (mean):', nested_score.mean())

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mdl = KernelRidge(kernel='rbf', alpha = grid.best_params_['reg__alpha'], gamma = grid.best_params_['reg__gamma']).fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    print('Test set (r2):', r2_score(y_test, y_pred))
    print('Test set (mse):', mean_squared_error(y_test, y_pred))
    print('Test set (mae):', mean_absolute_error(y_test, y_pred))


# In[25]:


phenos


# In[27]:


pheno = phenos[11]; print(pheno)
y = df.loc[:,pheno]


# In[28]:


metric = metrics[0]; print(metric)
X = df_z.filter(regex = metric)
run_krr(X,y)


# In[29]:


metric = metrics[3]; print(metric)
X = df_z.filter(regex = metric)
run_krr(X,y)


# In[30]:


metric = metrics[4]; print(metric)
X = df_z.filter(regex = metric)
run_krr(X,y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


if type(X) == pd.Series:
    X = X.values.reshape(-1,1)


# In[ ]:




