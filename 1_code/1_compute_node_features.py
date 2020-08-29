#!/usr/bin/env python
# coding: utf-8

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


sys.path.append('/Users/lindenmp/Google-Drive-Penn/work/research_projects/neurodev_cs_predictive/1_code/')
from func import set_proj_env, rank_int, node_strength, ave_control


# In[3]:


parc_str = 'schaefer'
parc_scale = 200
edge_weight = 'streamlineCount'
parcel_names, parcel_loc, drop_parcels, num_parcels = set_proj_env(parc_str = parc_str, parc_scale = parc_scale, edge_weight = edge_weight)


# In[4]:


# output file prefix
outfile_prefix = parc_str+'_'+str(parc_scale)+'_'+edge_weight+'_'
outfile_prefix


# ### Setup directory variables

# In[5]:


print(os.environ['PIPELINEDIR'])
if not os.path.exists(os.environ['PIPELINEDIR']): os.makedirs(os.environ['PIPELINEDIR'])


# In[6]:


storedir = os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'store')
print(storedir)
if not os.path.exists(storedir): os.makedirs(storedir)

outputdir = os.path.join(os.environ['PIPELINEDIR'], '1_compute_node_features', 'out')
print(outputdir)
if not os.path.exists(outputdir): os.makedirs(outputdir)


# In[7]:


figdir = os.path.join(os.environ['OUTPUTDIR'], 'figs')
print(figdir)
if not os.path.exists(figdir): os.makedirs(figdir)


# ## Load data

# In[8]:


# Load data
df = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '0_get_sample', 'out', 'df.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)
print(df.shape)


# In[9]:


# Missing data file for this subject only for schaefer 200
if parc_scale == 200:
    df.drop(labels = (112598, 5161), inplace=True)


# In[10]:


# output dataframe
str_labels = ['str_' + str(i) for i in range(num_parcels)]
ac_labels = ['ac_' + str(i) for i in range(num_parcels)]

df_node = pd.DataFrame(index = df.index, columns = str_labels + ac_labels)
print(df_node.shape)


# ## Load in structural connectivity matrices

# In[11]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[12]:


A = np.zeros((num_parcels, num_parcels, df.shape[0]))
for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['SC_NAME_TMP'].replace("scanid", str(index[1]))
    file_name = file_name.replace("bblid", str(index[0]))
    full_path = glob.glob(os.path.join(os.environ['SCDIR'], file_name))
    if i == 0: print(full_path)
    if len(full_path) > 0:
        mat_contents = sio.loadmat(full_path[0])
        a = mat_contents[os.environ['CONN_STR']]
        if parc_str == 'lausanne' and parc_variant == 'cortex_only':
            a = a[parcel_loc == 1,:]
            a = a[:,parcel_loc == 1]
        A[:,:,i] = a
    elif len(full_path) == 0:
        print(file_name + ': NOT FOUND')
        subj_filt[i] = True
        A[:,:,i] = np.full((num_parcels, num_parcels), np.nan)


# In[13]:


np.sum(subj_filt)


# In[14]:


if any(subj_filt):
    A = A[:,:,~subj_filt]
    df = df.loc[~subj_filt]
    df_node = df_node.loc[~subj_filt]
print(df_node.shape)


# ### Check if any subjects have disconnected nodes in A matrix

# In[15]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[16]:


for i in range(A.shape[2]):
    if np.any(np.sum(A[:,:,i], axis = 1) == 0):
        subj_filt[i] = True


# In[17]:


np.sum(subj_filt)


# In[18]:


if any(subj_filt):
    A = A[:,:,~subj_filt]
    df = df.loc[~subj_filt]
    df_node = df_node.loc[~subj_filt]
print(df_node.shape)


# In[19]:


np.sum(df['averageManualRating'] == 2)


# In[20]:


np.sum(df['dti64QAManualScore'] == 2)


# ### Get streamline count and network density

# In[21]:


A_c = np.zeros((A.shape[2],))
A_d = np.zeros((A.shape[2],))
for i in range(A.shape[2]):
    A_c[i] = np.sum(np.triu(A[:,:,i]))
    A_d[i] = np.count_nonzero(np.triu(A[:,:,i]))/((A[:,:,i].shape[0]**2-A[:,:,i].shape[0])/2)
df.loc[:,'streamline_count'] = A_c
df.loc[:,'network_density'] = A_d


# ### Compute node metrics

# In[22]:


# fc stored as 3d matrix, subjects of 3rd dim
S = np.zeros((df.shape[0], num_parcels))
AC = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    S[i,:] = node_strength(A[:,:,i])
    AC[i,:] = ave_control(A[:,:,i])

df_node.loc[:,str_labels] = S
df_node.loc[:,ac_labels] = AC


# ## Recalculate average control at different C params

# In[23]:


c_params = np.array([10, 100, 1000, 10000])
c_params


# In[24]:


# output dataframe
df_node_ac_overc = pd.DataFrame(index = df.index)

for c in c_params:
    print(c)
    ac_labels_new = ['ac_c' + str(c) + '_' + str(i) for i in range(num_parcels)]
    df_node_ac_temp = pd.DataFrame(index = df.index, columns = ac_labels_new)
    
    # fc stored as 3d matrix, subjects of 3rd dim
    AC = np.zeros((df.shape[0], num_parcels))
    for (i, (index, row)) in enumerate(df.iterrows()):
        AC[i,:] = ave_control(A[:,:,i], c = c)

    df_node_ac_temp.loc[:,ac_labels_new] = AC
    df_node_ac_overc = pd.concat((df_node_ac_overc, df_node_ac_temp), axis = 1)


# # Save out raw data

# In[25]:


print(df_node.isna().any().any())
print(df_node_ac_overc.isna().any().any())


# In[26]:


np.save(os.path.join(storedir, outfile_prefix+'A'), A)
df_node.to_csv(os.path.join(storedir, outfile_prefix+'df_node.csv'))
df_node_ac_overc.to_csv(os.path.join(storedir, outfile_prefix+'df_node_ac_overc.csv'))
df.to_csv(os.path.join(storedir, outfile_prefix+'df.csv'))


# # Export for prediction

# ## Normalize

# ### Covariates

# In[27]:


covs = ['ageAtScan1', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS', 'network_density', 'streamline_count']


# In[28]:


rank_r = np.zeros(len(covs),)

for i, cov in enumerate(covs):
    x = rank_int(df.loc[:,cov])
    rank_r[i] = sp.stats.spearmanr(df.loc[:,cov],x)[0]
    df.loc[:,cov] = x

print(np.sum(rank_r < 0.99))


# ### Node features

# In[29]:


rank_r = np.zeros(df_node.shape[1],)

for i, col in enumerate(df_node.columns):
    x = rank_int(df_node.loc[:,col])
    rank_r[i] = sp.stats.spearmanr(df_node.loc[:,col],x)[0]
    df_node.loc[:,col] = x

print(np.sum(rank_r < .99))


# In[30]:


rank_r = np.zeros(df_node_ac_overc.shape[1],)

for i, col in enumerate(df_node_ac_overc.columns):
    x = rank_int(df_node_ac_overc.loc[:,col])
    rank_r[i] = sp.stats.spearmanr(df_node_ac_overc.loc[:,col],x)[0]
    df_node_ac_overc.loc[:,col] = x

print(np.sum(rank_r < .99))


# ### Psychosis

# In[31]:


covs = ['ageAtScan1', 'sex', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS']
phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg']
print(phenos)

df_node.to_csv(os.path.join(outputdir, outfile_prefix+'X.csv'))
df_node_ac_overc.to_csv(os.path.join(outputdir, outfile_prefix+'X_ac_c.csv'))
df.loc[:,phenos].to_csv(os.path.join(outputdir, outfile_prefix+'y.csv'))
df.loc[:,covs].to_csv(os.path.join(outputdir, outfile_prefix+'c.csv'))

covs = ['ageAtScan1', 'sex', 'mprage_antsCT_vol_TBV', 'dti64MeanRelRMS', 'streamline_count']
df.loc[:,covs].to_csv(os.path.join(outputdir, outfile_prefix+'c_sc.csv'))

