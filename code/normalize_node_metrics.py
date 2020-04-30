#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[1]:


# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import nibabel as nib
import scipy.io as sio

# Stats
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import pingouin as pg

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


# In[2]:


sys.path.append('/Users/lindenmp/Dropbox/Work/ResProjects/neurodev_cs_predictive/code/func/')
from proj_environment import set_proj_env
sys.path.append('/Users/lindenmp/Dropbox/Work/git/pyfunc/')
from func import mark_outliers, winsorize_outliers_signed


# In[3]:


exclude_str = 't1Exclude'
extra_str = '' # '_vol_norm' '_noboxcox' '_consist'
edge_weight = 'streamlineCount' # 'streamlineCount' 'fa' 'mean_streamlineLength' 'adc'
parc_scale = 200
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(exclude_str = exclude_str,
                                                                                        parc_scale = parc_scale,
                                                                                       extra_str = extra_str, edge_weight = edge_weight)


# ### Setup output directory

# In[4]:


print(os.environ['MODELDIR'])
if not os.path.exists(os.environ['MODELDIR']): os.makedirs(os.environ['MODELDIR'])


# ### Data processing options

# In[5]:


wins_data = False
my_thresh = 3
norm_data = True


# ## Load data

# In[6]:


# Load data
df = pd.read_csv(os.path.join(os.environ['MODELDIR'], 'df_pheno.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)
print(df.shape)


# In[7]:


# Load data
df_node = pd.read_csv(os.path.join(os.environ['MODELDIR'], 'df_node_base.csv'))
df_node.set_index(['bblid', 'scanid'], inplace = True)
print(df_node.shape)


# In[8]:


df_node.head()


# In[9]:


metrics = ('vol', 'str', 'ac', 'mc')
dwi_metrics = ('str', 'ac', 'mc')
t1_metrics = ('vol',)


# In[10]:


for metric in metrics:
    x = df_node.filter(regex = metric, axis = 1).mean(axis = 1)
    outliers = mark_outliers(x, thresh = my_thresh)
    print(metric + ': ' + str(np.round((outliers.sum() / x.shape[0]) * 100,2)))


# A higher threshold than 3 probably makes sense but sticking with convention to avoid 'kick me' signs with reviewers.
# 
# Note, results are unaffected by setting my_thresh to 4,5, or 6

# ### Check frequency of outliers

# In[11]:


df_node_mask = pd.DataFrame(index = df_node.index, columns = df_node.columns)
for i, col in enumerate(df_node.columns):
    x = df_node.loc[:,col].copy()
    x_out = mark_outliers(x, thresh = my_thresh)
    df_node_mask.loc[:,col] = x_out

f, axes = plt.subplots(1,len(metrics))
f.set_figwidth(len(metrics)*5)
f.set_figheight(5)

for i, metric in enumerate(metrics):
    if df_node_mask.filter(regex = metric).sum().any():
        sns.distplot(df_node_mask.filter(regex = metric).sum()/df_node_mask.filter(regex = metric).shape[0]*100, ax = axes[i])


# ### Winsorize outliers

# In[12]:


my_str = '|'.join(dwi_metrics); print(my_str)
# my_str = 'ac|mc'


# In[13]:


if wins_data:
    for i, col in enumerate(df_node.filter(regex = my_str).columns):
        x = df_node.loc[:,col].copy()
        x_out = winsorize_outliers_signed(x, thresh = my_thresh)
        df_node.loc[:,col] = x_out
else:
    print('Skipping...')


# ## Normalize

# In[14]:


my_str = '|'.join(metrics); print(my_str)


# In[15]:


if np.any(df_node < 0):
    print('WARNING: some regional values are <0, box cox will fail')

if np.any(df_node == 0):
    print('WARNING: some regional values are == 0, box cox will fail')


# In[16]:


rank_r = np.zeros(df_node.filter(regex = my_str).shape[1])

# normalise
if norm_data:
    for i, col in enumerate(df_node.filter(regex = my_str).columns):
        # normalize regional metric
        x = sp.stats.boxcox(df_node.loc[:,col])[0]
        # check if rank order is preserved
        rank_r[i] = sp.stats.spearmanr(df_node.loc[:,col],x)[0]
        # store normalized version
        df_node.loc[:,col] = x
    print(np.sum(rank_r < .99))
else:
    print('Skipping...')


# ### Check distributions

# In[17]:


f, axes = plt.subplots(2,len(metrics))
f.set_figwidth(len(metrics)*5)
f.set_figheight(10)

for i, metric in enumerate(metrics):
    kur = np.zeros((df_node.filter(regex = metric).shape[1]))
    skew = np.zeros((df_node.filter(regex = metric).shape[1]))
    for j, node in enumerate(df_node.filter(regex = metric).columns):
        d = sp.stats.zscore(df_node.filter(regex = metric).loc[:,node])
        kur[j] = sp.stats.kurtosistest(d)[0]
        skew[j] = sp.stats.skewtest(d)[0]
    
    sns.distplot(kur, ax = axes[0,i])
    axes[0,i].set_xlabel(metric+': kurtosis')
    sns.distplot(skew, ax = axes[1,i])
    axes[1,i].set_xlabel(metric+': skewness')


# In[18]:


df_node.isna().any().any()


# ## Save out

# In[19]:


# Save out
df_node.to_csv(os.path.join(os.environ['MODELDIR'], 'df_node_normalized.csv'))
# df_node.to_csv(os.path.join(os.environ['MODELDIR'], 'df_node_normalized_my_thresh-'+str(my_thresh)+'.csv'))

