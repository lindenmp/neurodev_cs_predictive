#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[ ]:


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
from func import node_strength, ave_control, modal_control


# In[3]:


train_test_str = 'squeakycleanExclude'
exclude_str = 't1Exclude'
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(exclude_str = exclude_str, train_test_str = train_test_str)


# ### Setup output directory

# In[4]:


print(os.environ['MODELDIR'])
if not os.path.exists(os.environ['MODELDIR']): os.makedirs(os.environ['MODELDIR'])


# ## Load train/test .csv and setup node .csv

# In[5]:


os.path.join(os.environ['TRTEDIR'])


# In[6]:


# Load data
df = pd.read_csv(os.path.join(os.environ['TRTEDIR'], 'df_pheno.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)
print(df.shape)


# In[7]:


# # Missing data file for this subject only for schaefer 200
# if parc_str == 'schaefer' and parc_scale == 200:
#     df.drop(labels = (112598, 5161), inplace=True)


# In[8]:


# output dataframe
vol_labels = ['vol_' + str(i) for i in range(num_parcels)]
str_labels = ['str_' + str(i) for i in range(num_parcels)]
ac_labels = ['ac_' + str(i) for i in range(num_parcels)]
mc_labels = ['mc_' + str(i) for i in range(num_parcels)]

df_node = pd.DataFrame(index = df.index, columns = vol_labels + str_labels + ac_labels + mc_labels)
df_node.insert(0, train_test_str, df[train_test_str])

print(df_node.shape)


# ## Load in cortical volume

# In[9]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[10]:


VOL = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['VOL_NAME_TMP'].replace("bblid", str(index[0]))
    file_name = file_name.replace("scanid", str(index[1]))
    full_path = glob.glob(os.path.join(os.environ['VOLDIR'], file_name))
    if i == 0: print(full_path)    
    
    if len(full_path) > 0:
        img = nib.load(full_path[0])
        v = np.array(img.dataobj)
        v = v[v != 0]
        unique_elements, counts_elements = np.unique(v, return_counts=True)
        if len(unique_elements) == num_parcels:
            VOL[i,:] = counts_elements
        else:
            print(str(index) + '. Warning: not all parcels present')
            subj_filt[i] = True
    elif len(full_path) == 0:
        subj_filt[i] = True
    
df_node.loc[:,vol_labels] = VOL


# In[11]:


np.sum(subj_filt)


# In[12]:


if any(subj_filt):
    df = df.loc[~subj_filt]
    df_node = df_node.loc[~subj_filt]


# ## Load in structural connectivity matrices

# In[13]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[14]:


A = np.zeros((num_parcels, num_parcels, df.shape[0]))
for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['SC_NAME_TMP'].replace("scanid", str(index[1]))
    file_name = file_name.replace("bblid", str(index[0]))
    full_path = glob.glob(os.path.join(os.environ['SCDIR'], file_name))
    if i == 0: print(full_path)
    if len(full_path) > 0:
        mat_contents = sio.loadmat(full_path[0])
        a = mat_contents[os.environ['CONN_STR']]
        A[:,:,i] = a
    elif len(full_path) == 0:
        print(file_name + ': NOT FOUND')
        subj_filt[i] = True
        A[:,:,i] = np.full((num_parcels, num_parcels), np.nan)


# In[15]:


np.sum(subj_filt)


# In[16]:


if any(subj_filt):
    A = A[:,:,~subj_filt]
    df = df.loc[~subj_filt]
    df_node = df_node.loc[~subj_filt]
print(df_node.shape)


# ### Check if any subjects have disconnected nodes in A matrix

# In[17]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[18]:


for i in range(A.shape[2]):
    if np.any(np.sum(A[:,:,i], axis = 1) == 0):
        subj_filt[i] = True


# In[19]:


np.sum(subj_filt)


# In[20]:


if any(subj_filt):
    A = A[:,:,~subj_filt]
    df = df.loc[~subj_filt]
    df_node = df_node.loc[~subj_filt]
print(df_node.shape)


# ### Get streamline count and network density

# In[21]:


A_c = np.zeros((A.shape[2],))
A_d = np.zeros((A.shape[2],))
for i in range(A.shape[2]):
    A_c[i] = np.sum(np.triu(A[:,:,i]))
    A_d[i] = np.count_nonzero(np.triu(A[:,:,i]))/((A[:,:,i].shape[0]**2-A[:,:,i].shape[0])/2)
df.loc[:,'streamline_count'] = A_c
df.loc[:,'network_density'] = A_d


# ### Normalize A by regional volume

# In[22]:


vol_ref = os.path.join(os.environ['DERIVSDIR'], 'Schaefer2018_400_17Networks_PNC_2mm.nii.gz'))
img = nib.load(vol_ref)
v = np.array(img.dataobj)
v = v[v != 0]
unique_elements, counts_elements = np.unique(v, return_counts=True)


# In[23]:


for i in range(num_parcels):
    for j in range(num_parcels):
        denom = counts_elements[i] + counts_elements[j]
        A[i,j,:] = A[i,j,:]/denom


# ### Compute node metrics

# In[24]:


# fc stored as 3d matrix, subjects of 3rd dim
S = np.zeros((df.shape[0], num_parcels))
AC = np.zeros((df.shape[0], num_parcels))
MC = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    S[i,:] = node_strength(A[:,:,i])
    AC[i,:] = ave_control(A[:,:,i])
    MC[i,:] = modal_control(A[:,:,i])

df_node.loc[:,str_labels] = S
df_node.loc[:,ac_labels] = AC
df_node.loc[:,mc_labels] = MC


# ## Save out

# In[25]:


os.environ['MODELDIR']


# In[26]:


# Save out
np.save(os.path.join(os.environ['MODELDIR'], 'A'), A)
df_node.to_csv(os.path.join(os.environ['MODELDIR'], 'df_node_base.csv'))
df.to_csv(os.path.join(os.environ['MODELDIR'], 'df_pheno.csv'))

