#!/usr/bin/env python
# coding: utf-8

# # Preamble

# In[1]:


import os, sys, glob
import pandas as pd
import numpy as np
import scipy.io as sio
import nibabel as nib


# In[2]:


sys.path.append('/Users/lindenmp/Dropbox/Work/ResProjects/NormativeNeuroDev_CrossSec_DWI/code/func/')
from proj_environment import set_proj_env
from func import node_strength, ave_control, modal_control


# In[3]:


train_test_str = 'squeakycleanExclude'
exclude_str = 't1Exclude'
parc_str = 'schaefer'
parc_scale = 200
parcel_names, parcel_loc, drop_parcels, num_parcels, yeo_idx, yeo_labels = set_proj_env(train_test_str = train_test_str, exclude_str = exclude_str,
                                                                            parc_str = parc_str, parc_scale = parc_scale)


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


# Missing data file for this subject only for schaefer 200
if parc_str == 'schaefer' and parc_scale == 200:
    df.drop(labels = (112598, 5161), inplace=True)


# In[8]:


# output dataframe
ct_labels = ['ct_' + str(i) for i in range(num_parcels)]
vol_labels = ['vol_' + str(i) for i in range(num_parcels)]
str_labels = ['str_' + str(i) for i in range(num_parcels)]
ac_labels = ['ac_' + str(i) for i in range(num_parcels)]
mc_labels = ['mc_' + str(i) for i in range(num_parcels)]

df_node = pd.DataFrame(index = df.index, columns = ct_labels + vol_labels + str_labels + ac_labels + mc_labels)
df_node.insert(0, train_test_str, df[train_test_str])

print(df_node.shape)


# ## Load in cortical thickness and volume

# In[9]:


CT = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['CT_NAME_TMP'].replace("bblid", str(index[0]))
    file_name = file_name.replace("scanid", str(index[1]))
    full_path = glob.glob(os.path.join(os.environ['CTDIR'], file_name))
    if i == 0: print(full_path)    

    ct = np.loadtxt(full_path[0])
    CT[i,:] = ct
    
df_node.loc[:,ct_labels] = CT


# In[10]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[11]:


VOL = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = os.environ['VOL_NAME_TMP'].replace("bblid", str(index[0]))
    file_name = file_name.replace("scanid", str(index[1]))
    full_path = glob.glob(os.path.join(os.environ['VOLDIR'], file_name))
    if i == 0: print(full_path)    
    
    img = nib.load(full_path[0])
    v = np.array(img.dataobj)
    v = v[v != 0]
    unique_elements, counts_elements = np.unique(v, return_counts=True)
    if len(unique_elements) == num_parcels:
        VOL[i,:] = counts_elements
    else:
        print(str(index) + '. Warning: not all parcels present')
        subj_filt[i] = True
    
df_node.loc[:,vol_labels] = VOL


# In[12]:


np.sum(subj_filt)


# ## Load in connectivity matrices and compute node metrics

# In[13]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[14]:


# fc stored as 3d matrix, subjects of 3rd dim
A = np.zeros((num_parcels, num_parcels, df.shape[0]))
S = np.zeros((df.shape[0], num_parcels))
AC = np.zeros((df.shape[0], num_parcels))
MC = np.zeros((df.shape[0], num_parcels))

for (i, (index, row)) in enumerate(df.iterrows()):
    if parc_str == 'schaefer':
        file_name = os.environ['SC_NAME_TMP'].replace("scanid", str(index[1]))
        file_name = file_name.replace("bblid", str(index[0]))
        full_path = glob.glob(os.path.join(os.environ['SCDIR'], file_name))
        if i == 0: print(full_path)
        if len(full_path) > 0:
            mat_contents = sio.loadmat(full_path[0])
            a = mat_contents[os.environ['CONN_STR']]

            A[:,:,i] = a
            S[i,:] = node_strength(a)
            AC[i,:] = ave_control(a)
            MC[i,:] = modal_control(a)
        elif len(full_path) == 0:
            print(file_name + ': NOT FOUND')
            subj_filt[i] = True
            A[:,:,i] = np.full((num_parcels, num_parcels), np.nan)
            S[i,:] = np.full(num_parcels, np.nan)
            AC[i,:] = np.full(num_parcels, np.nan)
            MC[i,:] = np.full(num_parcels, np.nan)     

df_node.loc[:,str_labels] = S
df_node.loc[:,ac_labels] = AC
df_node.loc[:,mc_labels] = MC


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
    a = A[:,:,i]
    A_c[i] = np.sum(np.triu(a))
    A_d[i] = np.count_nonzero(np.triu(a))/((a.shape[0]**2-a.shape[0])/2)
df.loc[:,'streamline_count'] = A_c
df.loc[:,'network_density'] = A_d


# ## Save out

# In[22]:


os.environ['MODELDIR']


# In[23]:


# Save out
np.save(os.path.join(os.environ['MODELDIR'], 'A'), A)
df_node.to_csv(os.path.join(os.environ['MODELDIR'], 'df_node_base.csv'))
df.to_csv(os.path.join(os.environ['MODELDIR'], 'df_pheno.csv'))

